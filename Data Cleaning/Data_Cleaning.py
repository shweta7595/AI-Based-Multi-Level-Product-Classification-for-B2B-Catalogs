#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import re

# Read the Excel file
df = pd.read_excel('gemini_product_details_with_part_id.xlsx', engine='openpyxl', keep_default_na=False)
print("File loaded successfully.")

# List of brand name values to replace
brand_name_replacements = [
    'Brand Name:  Unknown (requires further investigation)',
    'Unbranded (or specify brand if known)',
    'Brand:  (Brand name needs to be determined from manufacturer information)',
    ' Brand Name:  Unspecified (Requires further information)',
    ' Brand:  [Brand Name Needed -  Information not provided]',
    ' Brand Name:  Not specified (requires further information)',
    'Dimensions unavailable'
]

# Replace the specified brand names with "Not Found"
if 'brand' in df.columns:
    for brand in brand_name_replacements:
        df['brand'] = df['brand'].replace(brand, 'Not Found')
else:
    print("Warning: Column 'brand' not found in the DataFrame.")


# Replace in valid values for Size with one set value N/A
size_replacements = [
    'N/A (Requires further information)',
    'Size or Dimensions: (Requires Manufacturer Information)',
    'Not specified, requires module specifications',
    'Unspecified',
    'Size or Dimensions:  Unspecified (Requires further information)',
    'N/A (Software)',
    'N/A (Software License)',
    'N/A (Service Contract)',
    'N/A (Software Service)',
    'Dimensions and weight not specified on page.',
    'Dimensions not specified on page, check compatibility with your system.',
    'Not applicable for this software subscription',
    'Not applicable (software subscription)',
    'Dimensions and weight not specified on page.',
    'Dimensions unavailable',
    'Not specified on the page.',
    'Not specified',
    'Dimensions and weight not listed on page.',
    'Not applicable',
    'Not Applicable',
    'N/A (Processor Chip)',
    'Dimensions unavailable without additional information',
    'N/A (Requires further specification)',
    'Dimensions unavailable without further information',
    '(Dimensions need to be specified)',
    'Dimensions Vary by Model (Specify Model Number for Exact Dimensions)',
    'N/A (requires further information)',
    '(Dimensions need to be specified, e.g., 10ft)',
    'Dimensions vary, check Google specs.',
    'Not specified, needs further information',
    'Requires further information',
    'Not applicable (Software License)',
    'N/A (Software subscription)',
    'Virtual Appliance (Software Download)',
    'Not specified on page',
    'Not Applicable (Processor)',
    'Not specified on the product page.',
    'N/A (Processor)',
    'N/A (Software Subscription)',
    'Virtual Appliance (no physical dimensions)',
    'Virtual Appliance (Download)',
    'Requires further information',
    'Dimensions not specified in available data.  Requires further information.',
    'Dimensions not specified in provided part number.  Further information required.',
    'Virtual Appliance (No Physical Dimensions)',
    'Virtual Appliance',
    'Unknown (requires further investigation)',
    'Virtual Appliance (Software Download)',
    'Not specified on the page, but standard RJ48 connector size.',
    'Not specified on page, typical headset dimensions',
    'Unknown (requires further information)','(Requires further information from manufacturer)','Dimensions unavailable (requires lookup)','(Requires further information)','N/A (Dimensions need clarification)','Dimensions not specified in prompt.  Requires further information.','(Requires lookup from HP documentation or other sources)','Dimensions not specified in prompt','Length:  (Requires further specification - needs manufacturer data)','Not Applicable (Software/Firmware)'
]

# Replace all matching entries with 'N/A'
df['size'] = df['size'].replace(size_replacements, 'N/A')


# Prompt: Replace specific text values in the 'size' column with 'Varies'
# List of values to replace
length_replacements = [
    'Varies (Specify cable length)',
    'N/A (Printhead, size varies by printer model)',
    'Variable (depending on specific cable type)',
    'Length varies depending on specific cable type (check HP documentation for exact dimensions)',
    'Dimensions Vary by Configuration (Requires Spec Sheet)',
    '(Requires further information)',
    'N/A (requires further information)',
    'Dimensions Vary by Configuration',
    'Dimensions vary by model (consult HP documentation)',
    'Length: Varies (requires further specification)',
    'Length: Varies (needs further specification)',
    'Dimensions vary by model. Consult HP documentation for specifics.',
    'Dimensions vary depending on specific cable configuration.  Requires further specification.',
    'Length varies (check HP specifications)',
    'Length varies (check specifications)','Dimensions vary, check Google specs.',''
    'Various lengths available (specify when ordering)',
    '(Dimensions need to be specified, e.g., 10ft)',
    'Dimensions vary, check Google specs.',
    'Varied (Specify length needed)',
    'Dimensions Vary by Model (Specify if known)',
    'Variable, see specifications',
    'Dimensions needed (Requires further information)',
    'Dimensions Vary per Cable'
]

# Columns to apply replacements
columns_to_check = ['size']  # Add other relevant columns if needed

# Replace matching values with 'Varies'
for col in columns_to_check:
    if col in df.columns:  # Ensure the column exists
        for replacement in length_replacements:
            df[col] = df[col].replace(replacement, 'Varies')
    else:
        print(f"Warning: Column '{col}' not found in the DataFrame.")


# Remove "(approx.)" or "(Approx.)" from 'size' only if value is not "N/A"
df['size'] = df['size'].apply(
    lambda x: re.sub(r'\(approx\.\)', '', str(x), flags=re.IGNORECASE)
)


# Filter rows where 'size' starts with any of the target phrases
match_conditions = df['size'].astype(str).str.startswith(
    ('Dimensions vary', 'Dimensions will vary', 'Varies depending','Length varies','Dimensions Vary')
)

# Display matching rows
matched_rows = df[match_conditions]

df.loc[match_conditions, 'size'] = 'Varies'


def remove_bracket_if_starts_with_example(text):
    if isinstance(text, str):
        # Find all bracketed expressions
        brackets = re.findall(r'\(.*?\)', text)
        for b in brackets:
            # Check if the content inside the bracket starts with 'example,'
            if b[1:].lower().startswith('example'):  # b[1:] removes the opening '('
                text = text.replace(b, '')  # Remove the full bracketed content
    return text

# Apply to the 'size' column
df['size'] = df['size'].apply(remove_bracket_if_starts_with_example)


def remove_parentheses(text):
  if isinstance(text, str):
    return re.sub(r'\([^)]*\)', '', text).strip()
  return text

# Apply the function to the 'brand' and 'size' columns
df['brand'] = df['brand'].apply(remove_parentheses)


# prompt: if brand starts with Brand Name: replace this 'Brand Name:' with ''
def clean_brand(product_title):
    if isinstance(product_title, str) and product_title.startswith('Product Title:'):
        return product_title.replace('Product Title:', '').strip()
    return product_title

df['product_title'] = df['product_title'].apply(clean_brand)

def clean_brand(brand_name):
    if isinstance(brand_name, str) and brand_name.startswith('Brand Name:'):
        return brand_name.replace('Brand Name:', '').strip()
    return brand_name

df['brand'] = df['brand'].apply(clean_brand)

def clean_brand(size):
    if isinstance(size, str) and size.startswith('Size or Dimensions:'):
        return size.replace('Size or Dimensions:', '').strip()
    return size

df['size'] = df['size'].apply(clean_brand)


def clean_brand(size):
    if isinstance(size, str) and size.startswith('Dimensions:'):
        return size.replace('Dimensions:', '').strip()
    return size

df['size'] = df['size'].apply(clean_brand)

def clean_brand(size):
    if isinstance(size, str) and size.startswith('Size:'):
        return size.replace('Size:', '').strip()
    return size

df['size'] = df['size'].apply(clean_brand)

def clean_brand(product_desc_1):
    if isinstance(product_desc_1, str) and product_desc_1.startswith('Short Product Description:'):
        return product_desc_1.replace('Short Product Description:', '').strip()
    return product_desc_1

df['product_desc_1'] = df['product_desc_1'].apply(clean_brand)

def clean_brand(technical_details):
    if isinstance(technical_details, str) and technical_details.startswith('Detailed Product Description:'):
        return technical_details.replace('Detailed Product Description:', '').strip()
    return technical_details

df['technical_details'] = df['technical_details'].apply(clean_brand)

def clean_brand(central_description):
    if isinstance(central_description, str) and central_description.startswith('Central Description:'):
        return central_description.replace('Central Description:', '').strip()
    return central_description

df['central_description'] = df['central_description'].apply(clean_brand)

def clean_brand(summary):
    if isinstance(summary, str) and summary.startswith('Summary:'):
        return summary.replace('Summary:', '').strip()
    return summary

df['summary'] = df['summary'].apply(clean_brand)


df.to_excel('cleaned_product_details_with_part_id.xlsx', index=False)
