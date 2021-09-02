
import pymysql
'''
mysql类
'''
class MysqlConnect():
    def __init__(self,host,port,user,passwd,database):
        self.host = host
        self.port = port
        self.user = user
        self.passwd = passwd
        self.database = database
        self.con = pymysql.connect(host=self.host, port=self.port, user=self.user,
                                 passwd=self.passwd, db=self.database, charset='utf8')
        self.cursor = self.con.cursor()

    def commitDB(self):
        self.con.commit()

    def closeDB(self):
        self.con.close()

    def execute_update_insert(self, sql_str):
        '''
        插入数据
        :param sql_str:
        :return:
        '''
        try:
            self.cursor.execute(sql_str)
        except Exception as e:
            print(e)

def testFun():
    host = '10.185.147.201'
    port = 31763
    user = 'cx_api_xsc-01'
    passwd = 'CX_api_xushicheng01#'
    db = 'hunandb'
    mydb = MysqlConnect(host, port, user, passwd, db)
    print('连接成功！！')
    mydb.closeDB()
testFun()
    # d1 = '1'
    # d2 = '2'
    # d3 = 'asd'
    # d4 = '12312d格式'
    # sql = "insert into nunan_query (query_key, query_value,query_keyword,query_value_keyword) VALUES ('" + d1 + "', '" + d2 + "', '" + d3 + "', '" + d4 + "')"
    # try:
    #     for _ in range(10006):
    #         mydb.execute_update_insert(sql)
    #         if _ % 200==0: #200提交一次，加快速度
    #             mydb.commitDB()
    # except Exception as ex:
    #     print(ex)
    # finally:
    #     mydb.commitDB()
    #     mydb.closeDB()



