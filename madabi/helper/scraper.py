from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

def get_article_text_with_selenium(url):
    # Setup Chrome options for headless browsing
    options = Options()
    options.headless = True
    options.add_argument("--headless=new")

    # Path to your chromedriver (update it as per your Chromedriver path)
    chrome_driver_path = '/Users/rshigapo/Downloads/chromedriver_mac64/chromedriver'

    # Initiate headless Chrome browser
    service = Service(chrome_driver_path, options=options)
    browser = webdriver.Chrome(service=service, options=options)

    browser.get(url)

    # Wait for the page to load (adjust time as needed)
    time.sleep(2)

    # Extract text (you need to update this part based on the specific website's structure)
    text = browser.find_element(By.TAG_NAME, "body").text
    # text = ' '.join([p.text for p in article_text])

    # Close the browser
    browser.quit()

    return text


# Example DOI
doi = 'https://doi.org/10.1016/j.respol.2023.104917'
text = get_article_text_with_selenium(doi)

# if article_text:
#    print(article_text)
# Data availability
# Supplemented

# https://github.com/quest-bih/oddpub/blob/master/R/ODDPub_functions.R
available = ["included", "deposited", "released", "is provided", "are provided",
             "contained in", "available", "reproduce", "accessible", "can be accessed",
             "submitted", "can be downloaded", "reported in", "uploaded", "are public on"]

was_available = ["was provided", "were provided", "was contained in",
                 "were contained in", "was available", "were available",
                 "was accessible", "were accessible", "deposited by", "were reproduced"]

not_available = ["not included", "not deposited", "not released",
                 "not provided", "not contained in", "not available",
                 "not accessible", "not submitted"]

repositories = ["figshare", "dryad", "zenodo", "dataverse", "DataverseNL",
                "osf", "open science framework", "mendeley data", "GIGADB",
                "GigaScience database", "OpenNeuro"]

data = ["data", "dataset", "datasets"]

source_code = ["source code", "analysis script", "github", "SAS script",
               "SPSS script", "R script", "R code", "python script",
               "python code", "matlab script", "matlab code"]

data_availability = ["Data sharing", "Data Availability Statement",
                     "Data Availability", "Data deposition", "Deposited Data",
                     "Data Archiving", "Availability of data and materials",
                     "Availability of data", "Data Accessibility",
                     "Accessibility of data"]

