import sys
sys.path.append('../')

def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')


def print_ok(*message):
    print('\033[92m', *message, '\033[0m')


def print_warning(*message):
    print('\033[93m', *message, '\033[0m')


def print_info(*message):
    print('\033[96m', *message, '\033[0m')

def print_blink(*message):
    print('\033[5;31;46m' + ' '.join(message) + '\033[0m')

