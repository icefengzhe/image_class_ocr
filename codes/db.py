# import pymysql
#
#
# class DB:
#     def __init__(self):
#         self.conn = pymysql.connect(host='localhost',
#                                     port=3306,
#                                     user='fengzhe',
#                                     passwd='00Fengzheup',
#                                     db='test'
#                                     )
#         self.cur = self.conn.cursor()
#         db = 'RES'
#         create_db = f"CREATE DATABASE IF NOT EXISTS {db}"
#         self.exec(create_db)
#         self.cur = self.cur.
#
#     def __del__(self):
#         self.cur.close()
#         self.conn.close()
#
#     def query(self, sql):
#         self.cur.execute(sql)
#         return self.cur.fetchall()
#
#     def exec(self, sql):
#         try:
#             self.cur.execute(sql)
#             self.conn.commit()
#         except Exception as e:
#             self.conn.rollback()
#             print(str(e))
#
#     def check_user(self, name):
#         result = self.query("select * from user where name='{}'".format(name))
#         return True if result else False
#
#     def del_user(self, name):
#         self.exec("delete from user where name='{}'".format(name))

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker
import pandas as pd


Base = declarative_base()

engine = create_engine('mysql+pymysql://cx_api_xsc-01:CX_api_xushicheng01#@10.185.147.201:31763/hunandb')


def result_to_db(result):
    dtype_dict = {'image': Text, 'type': String(20), 'content': Text}
    result.to_sql('image_class', engine, index=False, if_exists='append', dtype=dtype_dict)


if __name__ == '__main__':
    result = pd.DataFrame([['fengzhe', 'test', 'done']], columns=['image', 'type', 'content'])
    result_to_db(result)
