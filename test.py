import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

test = pd.read_csv("data/test.csv")
train = pd.read_csv("data/train.csv")
train.head()
train_split = train.drop(["Survived"], axis = 1)
combine = train_split.append(test)
# print(test.shape, train.shape)



# sex_pivot = train.pivot_table(index="Sex",values="Survived")
# sex_pivot.plot.bar()
# plt.show()

# class_pivot = train.pivot_table(index = "Pclass", values = "Survived")
# class_pivot.plot.bar()
# plt.show()

# print("Dimensions of train: {}".format(train.shape))
# print("Dimensions of test: {}".format(test.shape))
# print(train.head())

# print(train["Age"].describe())

# survived = train[train["Survived"]==1]
# died = train[train["Survived"] == 0]
# survived["Age"].plot.hist(alpha = 0.5, color = "red", bins = 50)
# died["Age"].plot.hist(alpha = 0.5, color = "blue", bins = 50)
# plt.legend(["Survived", "Died"])
# plt.show()

#phân loại tuổi

def process_age(data,cut_points,label_names):
    data["Age"] = data["Age"].fillna(-0.5)
    data["Age_categories"] = pd.cut(data["Age"],cut_points,labels=label_names)
    return data

cut_points = [-1,0,5,12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

combine = process_age(combine,cut_points,label_names)


# tìm và phân loại ra các loại giai cấp chức vụ 
titles = set()
for name in train['Name']:
    titles.add(name.split(',')[1].split('.')[0].strip())

Title_Dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

# thêm thuộc tính title để phân biệt giai cấp cho mỗi người

def get_titles(data):
    data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    data['Title'] = data.Title.map(Title_Dictionary)
    return data
combine = get_titles(combine)


# phân biệt cabin người đó ở 
def get_cabins(data):
    data['Cabin'] = data['Cabin'].map(lambda cabin:cabin[0])
    return data
combine["Cabin"] = train["Cabin"].fillna("U")
combine = get_cabins(combine)
#phân biệt ticket
def cleanTicket(ticket):
    ticket = ticket.replace('.','')
    ticket = ticket.replace('/','')
    ticket = ticket.split()
    ticket = map(lambda t : t.strip(), ticket)
    ticket = list(filter(lambda t : not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else: 
        return 'XXX'
combine['Ticket'] = combine['Ticket'].map(lambda t : cleanTicket(t))
#phân loại gia đình
combine['FamilySize'] = combine['Parch'] + combine['SibSp'] + 1
combine['Singleton'] = combine['FamilySize'].map(lambda s: 1 if s == 1 else 0)
combine['SmallFamily'] = combine['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
combine['LargeFamily'] = combine['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

#chuyển dữ liệu Convert categorical variable into dummy/indicator variables

def create_dummies(data,column_name):
    dummies = pd.get_dummies(data[column_name],prefix=column_name)
    data = pd.concat([data,dummies],axis=1)
    return data

for column in ["Pclass","Sex","Age_categories", "Cabin", "Embarked", "Title", "Ticket"]:
    combine = create_dummies(combine,column)
combine_1 = combine.drop(["FamilySize", "Name", "Sex",  "SibSp", "Parch", "Fare", "Ticket", "Cabin", "Embarked", "Age", "Pclass", "Title", "Age_categories"], axis = 1)
train_1 = combine_1.iloc[:891]
test_1 = combine_1.iloc[891:]

# model 
all_X = train_1
all_y = train['Survived']
#chia train và test để kiểm chứng
train_X, test_X, train_y, test_y = train_test_split(
    all_X, all_y, test_size=0.1,random_state=0)


# lr = LogisticRegression(penalty='l2')
# lr.fit(all_X, all_y)
# predictions = lr.predict(test_1)

forrest_params = dict(     
    max_depth = [n for n in range(9, 14)],     
    min_samples_split = [n for n in range(4, 11)], 
    min_samples_leaf = [n for n in range(2, 5)],     
    n_estimators = [n for n in range(10, 60, 10)],
)
forrest = RandomForestClassifier()
forest_cv = GridSearchCV(estimator=forrest, param_grid=forrest_params, cv=5)
forest_cv.fit(all_X, all_y)
predictions = forest_cv.predict(test_1)

test_ids = test_1["PassengerId"]
submission_df = {"PassengerId": test_ids,
                 "Survived": predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv("data/submission.csv",index=False)



