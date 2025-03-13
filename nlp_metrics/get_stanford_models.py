#!/usr/bin/env python
from __future__ import print_function
"""Imports: The necessary Python modules are imported:
os: Provides functions to interact with the operating system, such as file and directory manipulation.
shutil: Offers high-level file operations, including moving and removing files.
zipfile: A module to work with .zip archive files, enabling extraction of the Stanford CoreNLP package."""
import os
import shutil
import zipfile
"""his block ensures compatibility with both Python 2.x and Python 3.x:
In Python 3.x, urlretrieve is imported from urllib.request.
In Python 2.x, it is imported from urllib.
This enables the script to run on both Python versions, handling the file download in a platform-agnostic way.
"""
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

"""Variables:
core_nlp: This is the name of the Stanford CoreNLP zip archive that will be downloaded. It 
corresponds to the full package of CoreNLP models and tools.
spice_lib: The directory where the Stanford CoreNLP files and models will be placed.
jar: This pattern string defines the name of the jar file(s) (with and without the -models suffix) that 
are expected in the downloaded archive."""
def main():
    core_nlp = "stanford-corenlp-full-2015-12-09"
    spice_lib = "spice/lib"
    jar = "stanford-corenlp-3.6.0{}.jar"
    
    root_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
    
    jar_file = os.path.join(root_dir, spice_lib, jar)
    if os.path.exists(jar_file.format("")):
        print("Found Stanford CoreNLP.")
    else:
        print("Downloading Stanford CoreNLP...")
        core_nlp_zip = core_nlp + ".zip"
        urlretrieve(
            "http://nlp.stanford.edu/software/{}".format(core_nlp_zip),
            core_nlp_zip)
        print("Unzipping {}...".format(core_nlp_zip))
        zip_ref = zipfile.ZipFile(core_nlp_zip, "r")
        zip_ref.extractall(spice_lib)
        zip_ref.close()
        shutil.move(
            os.path.join(spice_lib, core_nlp, jar.format("")),
            spice_lib)
        shutil.move(
            os.path.join(spice_lib, core_nlp, jar.format("-models")),
            spice_lib)
        os.remove(core_nlp_zip)
        shutil.rmtree(os.path.join(spice_lib, core_nlp))
        print("Done.")
    

if __name__ == "__main__":
    main()

