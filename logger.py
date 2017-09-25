# =================================================================================
def logger_msg(prefix, message, num=None):
    if num is None:
        print(prefix + ': ' + message)
    else:
        print(prefix + '(' + num + '): ' + message)


# =================================================================================
def logger_error(message, num=None):
    logger_msg('!!! Error', message, num)


# =================================================================================
def logger_warning(message, num=None):
    logger_msg('>>> Warning', message, num)


# =================================================================================
def logger_info(message, num=None):
    logger_msg('... Info', message, num)


