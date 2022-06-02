from core.mysql_initial_build import *
from core.create_order import CreateOrder, DateEncoder
from core.update_order import UpdateOrder
from utils.logger import logger_config
from pprint import pprint
import datetime
import sqlalchemy
import json
import time


def build_database():
    CreateDatabase()  # 建立初始資料庫
    create_table()  # 建立初始資料表


def main():
    logs_name = "order_log"
    logger = logger_config().create_logger(logs_name)
    session = create_session()  # 建立指標

    CreateOrder.create_order_json()  # 建立測試的訂單資訊
    with open('./data/test_order_data.json', 'r', encoding="utf-8") as f:
        test_order_data = json.load(f)  # 讀取json訂單資訊

    # 將訂單資料寫入Order、OrderDetail表格中
    order_datas, order_id = CreateOrder(logs_name).order_data(test_order_data)
    order_datas_detail, detail_id = CreateOrder(logs_name).order_data_detail(test_order_data, order_id)

    # print(order_datas, order_datas_detail)
    # print(f"order_id:{order_id}")
    # print(f"detail_id:{detail_id}")

    # 調出最新一筆訂單資料，透過OrderId降冪排序找出最新一筆資料，並記錄最新一筆訂單狀態(last_result)
    last_result = session.query(CreateTableOrder).order_by(CreateTableOrder.OrderId.desc()).first()
    logger.info(f"訂單狀態:{last_result.MealStatus}")

    # 更新訂單狀態，若訂單為成立，將其更新為處理中；若訂單為處理中，將其更新為已結帳
    tmp_result = UpdateOrder(logs_name).update_to_processing(last_result)
    if tmp_result:  # 若訂單狀態有改變，則將暫存結果令為新的結果，若為False則不改變狀態
        last_result = tmp_result
    time.sleep(2)
    tmp_result = UpdateOrder(logs_name).update_to_done(last_result)
    if tmp_result:
        last_result = tmp_result

    # 儲存訂單的最終結果
    order_last_result = session.query(CreateTableOrder).filter_by(OrderId=order_id).first()
    order_last_result = order_last_result.__dict__
    del order_last_result['_sa_instance_state']
    with open("./data/order_last_result.json", 'w', encoding='utf-8') as f:
        json.dump(order_last_result, f, ensure_ascii=True, cls=DateEncoder)
    # 儲存訂單詳細資料的最終結果
    order_detail_last_result = session.query(CreateTableOrderDetail).filter_by(DetailId=detail_id).first()
    order_detail_last_result = order_detail_last_result.__dict__
    del order_detail_last_result['_sa_instance_state']
    with open("./data/order_detail_last_result.json", 'w', encoding='utf-8') as f:
        json.dump(order_detail_last_result, f, ensure_ascii=True, cls=DateEncoder)

    # 測試，刪除前一筆訂單資料
    delete_previous_order = False
    if delete_previous_order:  # filter_by替換成指定id也可以刪除指定訂單資料
        session.query(CreateTableOrder).filter_by(OrderId=order_id-1).delete()
        session.query(CreateTableOrderDetail).filter_by(DetailId=detail_id-1).delete()
        session.commit()

    session.close()  # 關閉指標


if __name__ == "__main__":
    # build_database()  # 建立資料庫
    # 主程式包含:讀取json格式資料，寫入資料庫的表格(Order、OrderDetail)，更新訂單狀態成立->處理中->已結帳
    # 儲存最終訂單結果，刪除前一筆訂單資料(預設為False)
    main()
