import datetime
import json
from core.mysql_initial_build import CreateTableOrder, CreateTableOrderDetail
from core.mysql_initial_build import create_session
from utils.logger import logger_config


class DateEncoder(json.JSONEncoder):
    # datetime日期格式轉換為json格式
    def default(self, obj):
        if isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return json.JSONEncoder.default(self, obj)


class CreateOrder:
    def __init__(self, logs_name):
        self.logs_name = logs_name
        self.logger = logger_config().create_logger(self.logs_name)

    @staticmethod
    def create_order_json():
        # 建立測試訂單.json
        datas = {
            # "OrderId": order_id,
            "OrderSubject": "測試訂單",
            "Customer": "大明",
            "MealStatus": "成立",
            "MealAmount": 240,
            "CreateTime": datetime.datetime.now(),
            "ModifiedTime": datetime.datetime.now(),
            # "DetailId": detail_id,
            "MealName": "便當",
            "MealPrice": 80,
            "MealCount": 3,
        }
        with open("./data/test_order_data.json", 'w', encoding='utf-8') as f:
            json.dump(datas, f, ensure_ascii=True, cls=DateEncoder)

    def order_data(self, test_order_data):
        # 根據test_order_data資料在資料庫的TableOrder表格建立訂單
        session = create_session()
        try:  # 讀取指定欄位對應的資料
            datas = {
                "OrderSubject": test_order_data["OrderSubject"],
                "Customer": test_order_data["Customer"],
                "MealStatus": test_order_data["MealStatus"],
                "MealAmount": test_order_data["MealAmount"],
                "CreateTime": test_order_data["CreateTime"],
                "ModifiedTime": test_order_data["ModifiedTime"]
            }
            session.add(CreateTableOrder(**datas))
            self.logger.info(f"訂單資訊:{datas}")
            session.commit()
            # 根據資料建立完訂單後查詢最新一筆訂單(剛剛建立的，並回傳order_id)
            result = session.query(CreateTableOrder).order_by(CreateTableOrder.OrderId.desc()).first()
            order_id = result.OrderId
            return datas, order_id
        except Exception as e:
            self.logger.error(e, exc_info=True)
            print(e.__class__.__name__)
        finally:
            session.close()

    def order_data_detail(self, test_order_data, order_id):
        # 根據test_order_data資料在資料庫的TableOrderDetail表格建立詳細訂單資訊
        session = create_session()
        try:
            datas = {
                # "DetailId": detail_id,  # 自動建立
                "OrderId": order_id,  # 根據前一個資料來賦值
                "MealName": test_order_data["MealName"],
                "MealPrice": test_order_data["MealPrice"],
                "MealCount": test_order_data["MealCount"],
                "CreateTime": test_order_data["CreateTime"],
                "ModifiedTime": test_order_data["ModifiedTime"]
            }
            session.add(CreateTableOrderDetail(**datas))
            self.logger.info(f"訂單詳細資訊:{datas}")
            session.commit()
            # 根據資料建立完訂單後查詢最新一筆訂單(剛剛建立的，並回傳detail_id)
            result = session.query(CreateTableOrderDetail).order_by(CreateTableOrderDetail.OrderId.desc()).first()
            detail_id = result.DetailId
            return datas, detail_id
        except Exception as e:
            self.logger.error(e, exc_info=True)
            print(e.__class__.__name__)
        finally:
            session.close()
