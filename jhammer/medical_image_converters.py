import os

import dicom2nifti as d2n


def nifti2dicom(input_file, output_dir, accession_number=1):
    """
    Require nifti2dicom.
    `sudo apt install nifti2dicom`
    https://github.com/biolab-unige/nifti2dicom
    Args:
        input_file (str or pathlib.Path):
        output_dir (str or pathlib.Path):
        accession_number (int, optional, default=1):

    Returns:

    """

    cmd = f"nifti2dicom -i {input_file} -o {output_dir} -a {accession_number}"
    result = os.popen(cmd)
    return result.readlines()


def dicom2nifti(input_dicom_series, output_nifti_file, pydicom_read_force=False):
    if pydicom_read_force:
        d2n.settings.pydicom_read_force = pydicom_read_force
    d2n.convert_directory(input_dicom_series, output_nifti_file)
