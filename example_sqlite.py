import os,sqlite3

db_file=os.path.join(os.path.dirname(__file__),'test.db')
if os.path.isfile(db_file):
    os.remove(db_file)

cnn=sqlite3.connect(db_file)
cursor=cnn.cursor()
cursor.execute('create table user(id varchar(10) primary key,name varchar(20),score int)')
cursor.execute(r"insert into user values ('A-001','Adam',95)")
cursor.execute(r"insert into user values ('A-002','Bob',78)")
cursor.execute(r"insert into user values ('A-003','Cindy',89)")
cursor.close()
cnn.commit()
cnn.close()

def get_score_in(low,high):
    cnn=sqlite3.connect(db_file)
    cursor=cnn.cursor()
    cursor.execute("select name from user where score >? and score<=? order by score asc" ,(low,high))
    values=cursor.fetchall()
    print(values)
    cursor.close()
    cnn.commit()
    cnn.close()

# assert get_score_in(80, 95) == ['Adam']
# assert get_score_in(60, 80) == ['Bob', 'Cindy']
# assert get_score_in(60, 100) == ['Cindy', 'Bob', 'Adam']
get_score_in(75,100)

print('Pass')