import mysql.connector

conn=mysql.connector.connect(user='',password='',database='test')
cursor=conn.cursor()
cursor.execute('create table user(id varchar(10) primary key,name varchar(20),score int)')
cursor.execute(r"insert into user values ('A-001','Adam',95)")
print(cursor.rowcount)
conn.commit()
cursor.close()
cursor.execute('select * from user where id=?',('1'))
values=cursor.fetchall()
print(values)
cursor.close()
conn.close()
