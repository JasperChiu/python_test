from core.mysql_initial_build import CreateTableOrder, CreateTableOrderDetail
from core.mysql_initial_build import create_session
from utils.logger import logger_config
import datetime


class UpdateOrder:
    def __init__(self, logs_name):
        self.logs_name = logs_name
        self.logger = logger_config().create_logger(self.logs_name)

    def update_to_processing(self, last_result):
        # 訂單狀態 從成立轉成處理中
        if last_result.MealStatus == "成立":
            session = create_session()  # 建立指標
            try:
                datas = {"MealStatus": "處理中",
                         "ModifiedTime": datetime.datetime.now()}
                # 更新訂單狀態為處理中，並更新修改日期
                session.query(CreateTableOrder).filter_by(OrderId=last_result.OrderId).update(datas)
                session.commit()
                last_result = session.query(CreateTableOrder).filter_by(OrderId=last_result.OrderId).first()
                self.logger.info(f"更新訂單狀態:{last_result.MealStatus}")
                self.logger.info(f"更新訂單修改日期:{last_result.ModifiedTime}")
                return last_result
            except Exception as e:
                self.logger.error(e, exc_info=True)
                return False
            finally:
                session.close()
        return False

    def update_to_done(self, last_result):
        # 訂單狀態 從處理中轉成已結帳
        if last_result.MealStatus == "處理中":
            session = create_session()  # 建立指標
            try:
                datas = {"MealStatus": "已結帳",
                         "ModifiedTime": datetime.datetime.now()}
                # 更新訂單狀態為處理中，並更新修改日期
                session.query(CreateTableOrder).filter_by(OrderId=last_result.OrderId).update(datas)
                session.commit()
                last_result = session.query(CreateTableOrder).filter_by(OrderId=last_result.OrderId).first()
                self.logger.info(f"更新訂單狀態:{last_result.MealStatus}")
                self.logger.info(f"更新訂單修改日期:{last_result.ModifiedTime}")
                return last_result
            except Exception as e:
                self.logger.error(e, exc_info=True)
                return False
            finally:
                session.close()
        return False
