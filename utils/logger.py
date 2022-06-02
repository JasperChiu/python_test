import logging
import os
from datetime import datetime


class logger_config:
    """
    如何使用logger.py
    首先在要套用本log格式的python檔案中，同路徑下要有這個logger.py檔案
    放在其他資料夾也可以，匯入時指向即可
    接下來在python檔案中引入logger.py檔案；
    $ from utils.logger import logger_config
    再來在 if __name__ == '__main__': 測試區塊中；
    $ logger = logger_config().create_logger('後綴檔名')
    會在logs資料夾下建立一個log檔案，其命名格式為  %Y-%m-%d_後綴檔名.log
    後綴檔名部分可以自行命名，在本舊書重製案例中是代入書名
    """

    def create_logger(self, file_name):
        """
        建立logger設定
        :param file_name: 設定log檔案的後綴檔名，引入定義時需命名
        :return:
        """
        dir_path = './logs/'  # 設定 logs 目錄
        filename = "{:%Y-%m-%d}".format(datetime.now()) + f'_{file_name}.log'  # 設定要儲存的檔名
        # config
        logging.captureWarnings(True)  # 捕捉 py waring message
        logger = logging.getLogger('Order')  # 設定logging名稱，可以自己改成要用的名稱
        logger.setLevel(logging.DEBUG)  # 設定層級
        detail_formatter = logging.Formatter("%(asctime)s-%(name)s-[%(filename)s-line:%(lineno)d]-"
                                             "%(funcName)s-%(levelname)s-%(message)s",
                                             datefmt="%Y%m%d %H:%M:%S")
        simple_formatter = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                                             datefmt="%Y%m%d %H:%M:%S")

        # 若不存在logs目錄則新建
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # 註記若要詳細資訊 logger.debug('debug message', exc_info=True)；後面要加註 exc_info=True
        # file handler；輸出log(使用detail_fmt)
        fileHandler = logging.FileHandler(f"./logs/{filename}", 'a', 'utf-8')  # 續寫設定
        # fileHandler = logging.FileHandler(f"./logs/{filename}", 'w', 'utf-8') # 覆寫設定
        fileHandler.setLevel(logging.DEBUG)
        fileHandler.setFormatter(detail_formatter)

        # console handler；輸出到控制台(使用simple_fmt)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setLevel(logging.DEBUG)
        consoleHandler.setFormatter(simple_formatter)

        # 如果已經例項過一個相同名字的 logger，則不用再追加 handler
        # 若是不添加此行，則會重複建立多個handlers，導致輸出重複的訊息
        if not logger.handlers:
            logger.addHandler(fileHandler)
            logger.addHandler(consoleHandler)
        return logger
