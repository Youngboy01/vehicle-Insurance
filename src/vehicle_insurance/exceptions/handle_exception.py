import sys
import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    detail extracted about file name, line number and error message
    :param error: Exception object
    :param error_detail: sys module to traceback details
    """
    _, _, exception_traceback = error_detail.exc_info()

    file_name = exception_traceback.tb_frame.f_code.co_filename
    line_number = exception_traceback.tb_lineno
    error_message = f"Error occurred in script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
    logging.error(error_message)
    return error_message


class VehicleInsuranceException(Exception):
    def __init__(self, error_message: Exception, error_detail: sys):
        """initializes the VehicleInsuranceException with error message and error details"""
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """returns the string representation of the error message"""
        return self.error_message
