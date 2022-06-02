import pymysql
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column
from sqlalchemy import Integer, String, DATETIME, Float
from sqlalchemy.orm import sessionmaker


Base = declarative_base()
# mysql需要設定使用者名稱:密碼@端口/資料庫名稱
# mysql+pymysql://<username>:<password>@<host>:<port>/<database_name>
engine_url = "mysql+pymysql://root:Jwander1098@127.0.0.1:3306/7000_python_test"
# 使用create_engine來連線，若將參數 echo 設為 True，會將所有執行的過程輸出到 cmd or terminal 上
engine = create_engine(engine_url, echo=False)


# 建立初始資料庫
class CreateDatabase:
    db_settings = {
        "host": "127.0.0.1",
        "port": 3306,
        "user": "root",
        "password": "Jwander1098",
        # "db": "7000_python_test",  # 資料庫名稱
        "charset": "utf8"
    }
    try:
        conn = pymysql.connect(**db_settings)  # 讀入字典的方式進入資料庫
        cursor = conn.cursor()  # 建立操作游標
        cursor.execute("commit")
        db_exist = cursor.execute("select * from information_schema.schemata where schema_name='7000_python_test'")
        if db_exist == 0:  # 若指定名稱db不存在則建立db
            cursor.execute("CREATE DATABASE 7000_python_test")
    except Exception as e:
        conn.rollback()
        print(e)
    finally:
        cursor.close()  # 關閉游標
        conn.close()  # 關閉連線

# 設定資料表結構 - TableOrder
class CreateTableOrder(Base):
    __tablename__ = "Order"
    # Column 建立一個欄位，此處設定OrderId為主鍵，並自動遞增
    OrderId = Column(Integer, primary_key=True, autoincrement=True)
    OrderSubject = Column(String(55))
    Customer = Column(String(55))
    MealStatus = Column(String(55))
    MealAmount = Column(Integer)
    CreateTime = Column(DATETIME)
    ModifiedTime = Column(DATETIME)

# 設定資料表結構 - TableOrderDetail
class CreateTableOrderDetail(Base):
    __tablename__ = "OrderDetail"
    DetailId = Column(Integer, primary_key=True, autoincrement=True)
    OrderId = Column(Integer)
    MealName = Column(String(55))
    MealPrice = Column(Integer)
    MealCount = Column(Integer)
    CreateTime = Column(DATETIME)
    ModifiedTime = Column(DATETIME)

# 建立資料表
def create_table():
    Base.metadata.create_all(engine)

# 刪除資料表
def drop_table():
    Base.metadata.drop_all(engine)

# 建立操作實體
def create_session():
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

if __name__ == '__main__':
    CreateDatabase()  # 建立初始資料庫
    create_table()
    # drop_table()
