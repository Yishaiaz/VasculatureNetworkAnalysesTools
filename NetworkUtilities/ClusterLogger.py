import os
import datetime


class CallableLogger:
    """
    CallableLogger is implementation for a callable logger, behaves exactly like print built-in but with a custom output stream (instead of console)
    Class attributes:
        stream_output_fpath - str
        log_entry_prefix - str
    Available methods:
         - returns
    """
    def __init__(self, stream_output_fpath: str, log_entry_prefix: str):
        self.stream_output_fpath = stream_output_fpath
        self.log_entry_prefix = log_entry_prefix

    def __call__(self, entry_raw_str:str, *args, **kwargs):
        now_str = datetime.datetime.now().strftime('%d%m%Y-%H:%M:%S')
        output_str = f"{now_str}|{self.log_entry_prefix}|{entry_raw_str}\n"
        with open(self.stream_output_fpath, 'a') as f:
            f.write(output_str)


if __name__ == '__main__':
    logger = CallableLogger('./here.txt', 'prefix')

    logger('this first line')
    logger('this second line')