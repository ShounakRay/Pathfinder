# @Author: Peter Guo <Guo>, Shounak Ray <Ray>
# @Date:   02-Mar-2021 12:03:30:304  GMT-0700
# @Email:  rijshouray@gmail.com
# @Filename: glencoe_s3.py
# @Last modified by:   Ray
# @Last modified time: 02-Mar-2021 17:03:42:429  GMT-0700
# @License: [Private IP]

# ENVIRONMENT:
# > Python 3.8.6

import io
import os
from datetime import datetime

import boto3
import pandas as pd
import pymssql
from cryptography.fernet import Fernet

"""########################################################################################
################################# ONLY FOR INTERNAL USE ###################################
########################################################################################"""
# ENCRYPTING THE KEYS SO THE KEY ITSELF DOES NOT APPEAR IN THE CODE
# >> IN CASE THIS FILE IS COMPROMISED DURING TRANSMISSION, OUR KEYS ARE SAFE
# >> THE FIRST PART OF THE KEY WILL BE PROVIDED BY WHITE WHALE VIA A DIFFERENT CHANNEL
total = Fernet.generate_key()
part_1, part_2 = (total[:int(len(total) / 2)], total[int(len(total) / 2):])
"""OUR INTERNAL KEYS: """
print('PART 1 OF INTERNAL KEY:\n\t' + str(part_1))
print('PART 2 OF INTERNAL KEY:\n\t' + str(part_2))
if not os.path.exists('Keys'):
    os.makedirs('Keys')
print('PART 1: ' + str(part_1).replace("b'", "").replace("'", ""),
      '\nPART 2: ' + str(part_2).replace("b'", "").replace("'", ""),
      file=open('Keys/internal_key_' + str(datetime.now()) + '.txt', 'w'))
# Add the two partial keys together to determine the complete key
# >> This is the key White Whale and our Client will use to encyrpt and decrypt the respective files
wwkey = part_1 + part_2
# Only run the line of code below of wwkey is string type and not byte type
# wwkey = bytes(wwkey.encode())
f = Fernet(wwkey)


def encrypt_msg(message):
    encrypted = f.encrypt(bytes(message.encode()))
    return encrypted


def decrypt_msg(encrypted_message):
    decrypted = f.decrypt(bytes(encrypted_message))
    return decrypted.decode()


decrypt_msg()


class DataStream():
    def __init__(self, bucket, access_key, secret_key):
        self.bucket = bucket
        self.client = boto3.client(
            's3',
            aws_access_key_id=str(decrypt_msg(access_key)),
            aws_secret_access_key=str(decrypt_msg(secret_key)),
        )

    def get_all_files(self, folder=''):
        paginator = self.client.get_paginator('list_objects')
        pages = paginator.paginate(Bucket=self.bucket, Prefix=folder)
        items = []
        try:
            for page in pages:
                for obj in page['Contents']:
                    items.append(obj['Key'])
            return items
        except Exception as e:
            return []

    def read_s3(self, path, header='infer', sheet_name=0):
        source = self.client.get_object(Bucket=self.bucket, Key=path)
        tdf = None
        if '.csv' in path:
            tdf = pd.read_csv(io.BytesIO(source['Body'].read()), encoding='cp1252', header=header)
        if '.xlsx' in path:
            tdf = pd.read_excel(io.BytesIO(source['Body'].read()),
                                encoding='cp1252', header=header, sheet_name=sheet_name)
        return tdf

    def upload_table(self, dataframe, file_name):
        csv_buffer = io.StringIO()
        dataframe.to_csv(csv_buffer)
        self.client.put_object(Bucket=self.bucket, Key=file_name, Body=csv_buffer.getvalue())

    def upload_data(self, file_name, upload_name=None):
        # If S3 object_name was not specified, use file_name
        if upload_name is None:
            upload_name = file_name
        # Upload the file
        try:
            response = self.client.upload_file(file_name, self.bucket, upload_name)
        except Exception as e:
            print(e)
            return 'Failed'
        return 'Success'


#####################################################################
# SET UP

access_key = bytes(''.encode())
secret_key = bytes(''.encode())
bucket = ""

DS = DataStream(bucket, access_key, secret_key)

# decrypt_msg(access_key)
# decrypt_msg(secret_key)
# encrypt_msg('some_key_from_AWS')

#####################################################################
# EXAMPLE 1: UPLOADING A DATAFRAME CREATED IN MEMORY TO S3
some_table = pd.DataFrame({"A": [i for i in range(0, 100)], "B": [i for i in range(100, 200)]})
DS.upload_table(some_table, 'test.csv')

# IN REALITY, YOU WOULD HAVE SOME QUERIES HERE PULL DATA FROM YOUR MSSQL-SERVER
# DOCUMENTATION: https://pymssql.readthedocs.io/en/stable/ref/pymssql.html
# BE SURE TO INSTALL ALL THE PACKAGES/PLUGINS FOR pymssql

# REPLACE THE FOLLOWING WITH YOUR OWN DB CREDENTIALS
server = "<localhost:1433>"
user = '<database_user>'
password = "<database_password>"
default_database_name = ""
conn = pymssql.connect(server, user, password, default_database_name)

# REPLACE WITH YOUR OWN QUERY, YOU MIGHT HAVE 'dbo.' IN FRONT OF table_name TO INDICATE SCHEMA
SQL_STMT = "SELECT * FROM table_name;"
data_table_123 = pd.read_sql(SQL_STMT, conn)

# DS.upload_table(data_table_123, 'name_of_this_data_table.csv')


#####################################################################
#### EXAMPLE 2: UPLOADING TABLES FROM LOCAL FILE DIRECTORY TO S3 ####

file_dir = "/path_to_the_folder/"
datafiles = os.listdir(file_dir)  # GET A LIST OF ALL FILES

# UPLOAD FILE TO S3
for file in datafiles:
    # PASS IN THE FULL FILE PATH
    file_path = file_dir + file

    # FILES WITH THE SAME NAME WILL OVER OVER-WRITTEN
    status = DS.upload_data("file_path", file)
    print(file, "upload status:", status)


#####################################################################
# GET A LIST OF ALL FILES IN THE S3 BUCKET
all_files_in_s3_bucket = DS.get_all_files()

# READ FILE FROM BOARDWALK S3 BUCKET
data_table_downloaded_from_s3 = DS.read_s3("test.csv")
