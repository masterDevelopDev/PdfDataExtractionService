from extract_pdf_file import PdfExtractor
from PIL import Image
import re
import boto3
import json
import torch
import numpy as np
import clip
import io

# Initialize AWS S3 client to interact with S3 buckets.
s3 = boto3.client('s3')
# Set device to GPU if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, preprocess = clip.load('ViT-B/32', device=device)
model.eval()

def model_fn(model_dir):
    """
    Load and prepare the CLIP model.

    This function is invoked at the start in a SageMaker environment to load
    the model and its preprocessing pipeline. The model is set to evaluation mode.

    Args:
    - model_dir (str): Directory where the model is stored.

    Returns:
    - dict: Dictionary containing the model and preprocessing model.
    """
    return {"model": model, "preprocess": preprocess}


def input_fn(request_body, request_content_type):
    """
    Process the input data for model prediction.

    This function decodes the request body, extracts relevant data from JSON,
    and uses a PDF extractor to retrieve text and images from a PDF file.

    Args:
    - request_body (bytes): Encoded data from the request.
    - request_content_type (str): Type of the request.

    Returns:
    - Tuple: Combined text, list of image keys, list of images, and the output bucket name.
    """
    print(f'The request body is: {request_body}')
    data_str = request_body.decode('utf-8')
    # Parse the string as a JSON object
    data_dict = json.loads(data_str)
    print(f'This is what data_dict looks like: {data_dict}')
    output_bucket = data_dict['OutputImageBucket']
    base64_file = data_dict['Base64File']
    pdf_name = re.sub(r'\.pdf$', '', data_dict['Key'])
    print(f'The pdf_name is: {pdf_name}')
    print(f'Extracting the data from the pdf file...')
    pdf_extractor = PdfExtractor(pdf_name, base64_file, output_bucket)
    print(f'The data from the pdf file has been extracted.')
    combined_text, list_image_keys, list_images = pdf_extractor.extract_data()
    return combined_text, list_image_keys, list_images, output_bucket


def predict_fn(input_data, model_artifacts):
    """
    Generate predictions using the model.

    This function preprocesses images, loads the CLIP model, and generates image embeddings
    and returns them with extracted text.

    Args:
    - input_data (tuple): Extracted text, image keys, images, and output bucket from input_fn.
    - model_artifacts (dict): Loaded model artifacts.

    Returns:
    - dict: Dictionary containing full text and image embeddings.
    """
    combined_text, list_image_keys, list_images, output_bucket = input_data
    print(f'This is the combined text: {combined_text}')
    print(f'This is the list_image_keys: {list_image_keys}')
    print(f'This is the output_bucket: {output_bucket}')
    print(f'Generating the list of the preprocess images...')
    list_preprocessed_images = [preprocess(image) for image in list_images]
    print(f'The list of the preprocess images has been generated.')
    preprocessed_images_input = torch.tensor(np.stack(list_preprocessed_images))
    preprocessed_images_input = preprocessed_images_input.to(device)
    print(f"This is what preprocessed_images_input looks like: {preprocessed_images_input}")
    print("Generating prediction.")
    with torch.no_grad():
        print(f"Computing the embeddings ...")
        images_features = model.encode_image(preprocessed_images_input).float()
        print(f"The embeddings have been computed.")
    image_features_list = [{'key': list_image_keys[idx],
                            'embedding': images_features[idx].tolist()} for idx in range(len(list_image_keys))]
    print(f"The features list looks like this: {image_features_list}")
    return {'full_text': combined_text, 'images': image_features_list}


def output_fn(prediction_output, accept='application/json'):
    """
    Serialize the prediction output.

    Converts the prediction output into the specified format based on the 'accept' argument.

    Args:
    - prediction_output (dict): Output from the predict function.
    - accept (str): Desired output format.

    Returns:
    - bytes or str: Serialized prediction output.
    """
    print("Serializing the generated output.")
    print(f"The accept looks like this: {accept}")
    print(f"This is what the prediction output looks like: {prediction_output}")
    if accept == 'application/x-npy':
        output = np.array(prediction_output)
        buffer = io.BytesIO()
        np.save(buffer, output)
        buffer.seek(0)
        return buffer.getvalue()
    elif accept == 'application/json':
        return json.dumps(prediction_output)
    else:
        raise ValueError(f'Unsupported accept type: {accept}')