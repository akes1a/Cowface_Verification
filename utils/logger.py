def setup_logger(name, log_file=None):
    """设置日志记录器"""
    import logging
    logger = logging.getLogger(name)
    # 配置日志格式、级别等
    return logger