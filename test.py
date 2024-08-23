import sys

from dropbox.client import DropboxClient

# get an access token, local (from) directory, and Dropbox (to) directory

# from the command-line

access_token, local_directory, dropbox_destination = sys.argv[1:4]

client = DropboxClient(access_token)

# enumerate local files recursively

for root, dirs, files in os.walk(local_directory):

for filename in files:

# construct the full local path

local_path = os.path.join(root, filename)

# construct the full Dropbox path

relative_path = os.path.relpath(local_path, local_directory)

dropbox_path = os.path.join(dropbox_destination, relative_path)

# upload the file

with open(local_path, 'rb') as f:

client.put_file(dropbox_path, f)