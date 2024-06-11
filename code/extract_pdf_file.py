import boto3
import io
import base64
from pypdf import PdfReader
import numpy as np
from PIL import Image
import easyocr
import pypdf
import logging
from clean_text import *

# Configure basic logging to display information messages.
logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)

# Initialize AWS S3 client for S3 operations.
s3_client = boto3.client('s3')

# Initialize EasyOCR reader with English and Simplified Chinese, utilizing GPU if available.
reader = easyocr.Reader(['en', 'ch_sim'], gpu=True)

# Constants for text extraction and image processing.
EXTRACTED_TEXT_FROM_IMAGE = 'extracted_text_from_images'
EXTRACTED_TEXT_FROM_PDF = 'extracted_text_from_pdf'
COMBINED_TEXT = 'combined_text'
AVG_PIXEL_VALUE = 50
THRESHOLD_WHITE_PIXEL = 254
THRESHOLD_BLACK_PIXEL = 3

class PdfExtractor:
    def __init__(self, pdf_name, base64_file, output_bucket):
        """
        Initialize the PdfExtractor with the PDF file details.

        Args:
        - pdf_name (str): Name of the PDF file.
        - base64_file (str): Base64 encoded string of the PDF file.
        - output_bucket (str): Name of the S3 bucket for output storage.
        """
        LOG.info("Initializing PdfExtractor")
        self.pdf_name = pdf_name
        self.pdf_file = io.BytesIO(base64.b64decode(base64_file))
        self.output_bucket = output_bucket


    def extract_data(self):
        """
        Extract both text and images from the PDF.

        This method extracts text directly from the PDF and via OCR from images within the PDF.
        It then combines these texts and handles any exceptions during the process.

        Returns:
        - tuple: Combined text, list of image keys, and list of Image objects.
        """
        try:
            list_image_keys, list_images, extracted_text_from_image = PdfExtractor.extract_images_text_from_pdf(self.pdf_file, self.pdf_name, self.output_bucket)
            extracted_text_from_pdf = PdfExtractor.extract_text_from_pdf(self.pdf_file)
            combined_text = extracted_text_from_pdf + extracted_text_from_image
            combined_text = combined_text.replace('\n', ' ')
            return combined_text, list_image_keys, list_images
        except Exception as e:
            print(
                f"An error occurred: {e} - the file located in s3://{self.input_bucket}/{self.pdf_name}.pdf has not been processed.")
        

    @staticmethod
    def extract_images_text_from_pdf(pdf_file, pdf_name, output_bucket):
        """
        Extract text from images within a PDF file.

        This static method opens the PDF, iterates through each image, performs OCR,
        cleans the text, and uploads the images to S3.

        Args:
        - pdf_file (io.BytesIO): Byte stream of the PDF file.
        - pdf_name (str): Name of the PDF file.
        - output_bucket (str): S3 bucket name for saving images.

        Returns:
        - tuple: List of image keys, list of Image objects, and extracted text from images.
        """
        pdf = PdfReader(pdf_file)
        text_buffer = io.StringIO()
        list_image_keys = []
        list_images = []
        for page_index, page in enumerate(pdf.pages, start=1):
            for image_index, image_file in enumerate(page.images, start=1):
                image = Image.open(io.BytesIO(image_file)).convert('RGB')
                if THRESHOLD_WHITE_PIXEL > np.mean(image) > THRESHOLD_BLACK_PIXEL:
                    result = reader.readtext(np.array(image))
                    text = ' '.join([detected_text[1] for detected_text in result])
                    cleaner_text = TextCleaner(text)
                    text_buffer.write(cleaner_text.clean() + ' ')
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    image_key = f"{pdf_name}_{page_index}_{image_index}.png"
                    list_image_keys.append(image_key)
                    list_images.append(image)
                    s3_client.upload_fileobj(img_byte_arr, Bucket=output_bucket, Key=image_key)
        extracted_text_from_image = text_buffer.getvalue()
        text_buffer.close()
        return list_image_keys, list_images, extracted_text_from_image

    @staticmethod
    def extract_text_from_pdf(pdf_file):
        """
        Extract text directly from a PDF file.

        This static method uses PyPDF to read the PDF and extract text from each page.

        Args:
        - pdf_file (io.BytesIO): Byte stream of the PDF file.

        Returns:
        - str: Extracted text from the PDF.
        """
        pdf_reader = pypdf.PdfReader(pdf_file)
        num_pages = len(pdf_reader.pages)
        text_buffer = io.StringIO()
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            cleaner_text = TextCleaner(text)  
            text_buffer.write(cleaner_text.clean())
        text_buffer.seek(0)
        extracted_text_from_pdf = text_buffer.getvalue()
        return extracted_text_from_pdf
    