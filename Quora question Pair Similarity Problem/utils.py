import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score, log_loss
import math
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV





# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)

    A = (((C.T) / (C.sum(axis=1))).T)

    B = (C / C.sum(axis=0))
    num_cls = len(set(test_y))

    labels = list(range(0,num_cls))
    # representing A in heatmap format
    print("-" * 20, "Confusion matrix", "-" * 20)
    plt.figure(figsize=(7, 3))
    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    print("-" * 20, "Precision matrix (Columm Sum=1)", "-" * 20)
    plt.figure(figsize=(7, 3))
    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

    # representing B in heatmap format
    print("-" * 20, "Recall matrix (Row sum=1)", "-" * 20)
    plt.figure(figsize=(7, 3))
    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.show()

def build_random_model(test_df, cv_df, y_cv, y_test):
    # we need to generate 9 numbers and the sum of numbers should be 1
    # one solution is to genarate 9 numbers and divide each of the numbers by their sum
    # ref: https://stackoverflow.com/a/18662466/4084039
    num_cls = len(set(y_test))
    test_data_len = test_df.shape[0]
    cv_data_len = cv_df.shape[0]

    # we create a output array that has exactly same size as the CV data
    cv_predicted_y = np.zeros((cv_data_len, num_cls))
    for i in range(cv_data_len):
        rand_probs = np.random.rand(1, num_cls)
        cv_predicted_y[i] = ((rand_probs / sum(sum(rand_probs)))[0])
    print("Log loss on Cross Validation Data using Random Model", log_loss(y_cv, cv_predicted_y, eps=1e-15))

    # Test-Set error.
    # we create a output array that has exactly same as the test data
    test_predicted_y = np.zeros((test_data_len, num_cls))
    for i in range(test_data_len):
        rand_probs = np.random.rand(1, num_cls)
        test_predicted_y[i] = ((rand_probs / sum(sum(rand_probs)))[0])
    print("Log loss on Test Data using Random Model", log_loss(y_test, test_predicted_y, eps=1e-15))

    predicted_y = np.argmax(test_predicted_y, axis=1)
    return predicted_y

def extract_dictionary_paddle(cls_text, col_name):
    dictionary = defaultdict(int)
    for index, row in cls_text.iterrows():
        for word in row[col_name].split():
            dictionary[word] += 1
    return dictionary

def get_totaldict_and_dictlist(df, col_name, target_class, no_classs):
    dict_list = []
    # dict_list =[] contains 9 dictoinaries each corresponds to a class
    for i in range(1, no_classs+1):
        cls_text = df[df[target_class] == i]
        # build a word dict based on the words in that class
        dict_list.append(extract_dictionary_paddle(cls_text, col_name))
        # append it to dict_list

    # dict_list[i] is build on i'th  class text data
    # total_dict is buid on whole training text data
    total_dict = extract_dictionary_paddle(df, col_name)
    return dict_list, total_dict

# def get_feature_name_and_vector(train_df, col_name):
#     # building a CountVectorizer with all the words that occured minimum 3 times in train data
#     text_vectorizer = CountVectorizer(min_df=3)
#     train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df[col_name])
#     # getting all the feature names (words)
#     train_text_features = text_vectorizer.get_feature_names()
#
#     # train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector
#     train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1
#
#     # zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured
#     text_fea_dict = dict(zip(list(train_text_features), train_text_fea_counts))
#
#     print("Total number of unique words in train data :", len(train_text_features))
#     return text_fea_dict, train_text_features


# https://stackoverflow.com/a/1602964
def get_text_responsecoding(df, col_name, target_class, no_classs=2):
    dict_list, total_dict = get_totaldict_and_dictlist(df, col_name, target_class, no_classs)
    # text_fea_dict, train_text_features = self.get_feature_name_and_vector(df, col_name)

    text_feature_responseCoding = np.zeros((df.shape[0], no_classs))
    for i in range(0, no_classs):
        row_index = 0
        for index, row in df.iterrows():
            sum_prob = 0
            for word in row[col_name].split():
                sum_prob += math.log(((dict_list[i].get(word, 0) + 10) / (total_dict.get(word, 0) + 90)))
            text_feature_responseCoding[row_index][i] = math.exp(sum_prob / len(row[col_name].split()))
            row_index += 1
    return text_feature_responseCoding


def get_gv_fea_dict(alpha, feature, df):
    # value_count: it contains a dict like
    # print(train_df['Gene'].value_counts())
    # output:
    #        {BRCA1      174
    #         TP53       106
    #         EGFR        86
    #         BRCA2       75
    #         PTEN        69
    #         KIT         61
    #         BRAF        60
    #         ERBB2       47
    #         PDGFRA      46
    #         ...}
    # print(train_df['Variation'].value_counts())
    # output:
    # {
    # Truncating_Mutations                     63
    # Deletion                                 43
    # Amplification                            43
    # Fusions                                  22
    # Overexpression                            3
    # E17K                                      3
    # Q61L                                      3
    # S222D                                     2
    # P130S                                     2
    # ...
    # }
    value_count = df[feature].value_counts()

    # gv_dict : Gene Variation Dict, which contains the probability array for each gene/variation
    gv_dict = dict()

    # denominator will contain the number of time that particular feature occured in whole data
    for i, denominator in value_count.items():
        # vec will contain (p(yi==1/Gi) probability of gene/variation belongs to perticular class
        # vec is 9 diamensional vector
        vec = []
        for k in range(1, 10):
            # print(train_df.loc[(train_df['Class']==1) & (train_df['Gene']=='BRCA1')])
            #         ID   Gene             Variation  Class
            # 2470  2470  BRCA1                S1715C      1
            # 2486  2486  BRCA1                S1841R      1
            # 2614  2614  BRCA1                   M1R      1
            # 2432  2432  BRCA1                L1657P      1
            # 2567  2567  BRCA1                T1685A      1
            # 2583  2583  BRCA1                E1660G      1
            # 2634  2634  BRCA1                W1718L      1
            # cls_cnt.shape[0] will return the number of rows

            cls_cnt = df.loc[(df['Class'] == k) & (df[feature] == i)]

            # cls_cnt.shape[0](numerator) will contain the number of time that particular feature occured in whole data
            vec.append((cls_cnt.shape[0] + alpha * 10) / (denominator + 90 * alpha))

        # we are adding the gene/variation to the dict as key and vec as value
        gv_dict[i] = vec
    return gv_dict


# Get Gene variation feature
def get_gv_feature(alpha, feature, df):
    # print(gv_dict)
    #     {'BRCA1': [0.20075757575757575, 0.03787878787878788, 0.068181818181818177, 0.13636363636363635, 0.25, 0.19318181818181818, 0.03787878787878788, 0.03787878787878788, 0.03787878787878788],
    #      'TP53': [0.32142857142857145, 0.061224489795918366, 0.061224489795918366, 0.27040816326530615, 0.061224489795918366, 0.066326530612244902, 0.051020408163265307, 0.051020408163265307, 0.056122448979591837],
    #      'EGFR': [0.056818181818181816, 0.21590909090909091, 0.0625, 0.068181818181818177, 0.068181818181818177, 0.0625, 0.34659090909090912, 0.0625, 0.056818181818181816],
    #      'BRCA2': [0.13333333333333333, 0.060606060606060608, 0.060606060606060608, 0.078787878787878782, 0.1393939393939394, 0.34545454545454546, 0.060606060606060608, 0.060606060606060608, 0.060606060606060608],
    #      'PTEN': [0.069182389937106917, 0.062893081761006289, 0.069182389937106917, 0.46540880503144655, 0.075471698113207544, 0.062893081761006289, 0.069182389937106917, 0.062893081761006289, 0.062893081761006289],
    #      'KIT': [0.066225165562913912, 0.25165562913907286, 0.072847682119205295, 0.072847682119205295, 0.066225165562913912, 0.066225165562913912, 0.27152317880794702, 0.066225165562913912, 0.066225165562913912],
    #      'BRAF': [0.066666666666666666, 0.17999999999999999, 0.073333333333333334, 0.073333333333333334, 0.093333333333333338, 0.080000000000000002, 0.29999999999999999, 0.066666666666666666, 0.066666666666666666],
    #      ...
    #     }
    gv_dict = get_gv_fea_dict(alpha, feature, df)
    # value_count is similar in get_gv_fea_dict
    value_count = df[feature].value_counts()

    # gv_fea: Gene_variation feature, it will contain the feature for each feature value in the data
    gv_fea = []
    # for every feature values in the given data frame we will check if it is there in the train data then we will add the feature to gv_fea
    # if not we will add [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9] to gv_fea
    for index, row in df.iterrows():
        if row[feature] in dict(value_count).keys():
            gv_fea.append(gv_dict[row[feature]])
        else:
            gv_fea.append([1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9])
    #             gv_fea.append([-1,-1,-1,-1,-1,-1,-1,-1,-1])
    return gv_fea

### Function used in ML models
def predict_and_plot_confusion_matrix(train_x, train_y,test_x, test_y, clf, show_plot = None):
    clf.fit(train_x, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x, train_y)
    pred_y = sig_clf.predict(test_x)

    # for calculating log_loss we willl provide the array of probabilities belongs to each class
    print("Log loss :",log_loss(test_y, sig_clf.predict_proba(test_x)))
    # calculating the number of data points that are misclassified
    mis_classified_points = np.count_nonzero((pred_y- test_y))/test_y.shape[0]
    print("Number of mis-classified points :", mis_classified_points)
    if show_plot:
        plot_confusion_matrix(test_y, pred_y)
    return mis_classified_points

def report_log_loss(train_x, train_y, test_x, test_y,  clf):
    clf.fit(train_x, train_y)
    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")
    sig_clf.fit(train_x, train_y)
    sig_clf_probs = sig_clf.predict_proba(test_x)
    return log_loss(test_y, sig_clf_probs, eps=1e-15)


# this function will be used just for naive bayes
# for the given indices, we will print the name of the features
# and we will check whether the feature present in the test point text or not
