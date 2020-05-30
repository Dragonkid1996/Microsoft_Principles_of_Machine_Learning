import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math

auto_prices = pd.read_csv('Automobile price data _Raw_.csv')
auto_prices.head(20)

auto_prices.columns = [str.replace('-', '_') for str in auto_prices.columns]

print((auto_prices.astype(np.object) == '?').any())
print(auto_prices.dtypes)

for col in auto_prices.columns:
    if auto_prices[col].dtype == object:
        count = 0
        count = [count + 1 for x in auto_prices[col] if x == '?']
        print(col + ' ' + str(sum(count)))

#%%
# DROP COLUMN WITH TOO MANY MISSING VALUES
auto_prices.drop('normalized_losses', axis = 1, inplace = True)
# REMOVE ROWS WITH MISSING VALUES, ACCOUNTING FOR MISSING VALUES CODED AS '?'
cols = ['price', 'bore', 'stroke',
        'horsepower', 'peak_rpm']
for column in cols:
    auto_prices.loc[auto_prices[column] == '?', column] = np.nan
auto_prices.dropna(axis = 0, inplace = True)
auto_prices.shape

#%%
# TRANSFORM COLUMN DATA TYPE
for column in cols:
    auto_prices[column] = pd.to_numeric(auto_prices[column])
print(auto_prices[cols].dtypes)

#%%
#AGGREGATING CATEGORICAL VARIABLES
auto_prices['num_of_cylinders'].value_counts()
cylinder_categories = {'three':'three_four', 'four':'three_four',
                       'five':'five_six', 'six':'five_six',
                       'eight':'eight_twelve','twelve':'eight_twelve'}
auto_prices['num_of_cylinders'] = [cylinder_categories[x] for x in auto_prices['num_of_cylinders']]
auto_prices['num_of_cylinders'].value_counts()

#MAKE A BOX PLOT
def plot_box(auto_prices, col, col_y = 'price'):
    sns.set_style("whitegrid")
    sns.boxplot(col, col_y, data=auto_prices)
    plt.xlabel(col)
    plt.ylabel(col_y)
    plt.show()
    
plot_box(auto_prices, 'num_of_cylinders')

#%%
auto_prices['body_style'].value_counts()

body_cats = {'sedan':'sedan', 'hatchback':'hatchback', 'wagon':'wagon',
             'hardtop':'hardtop_convert', 'convertible':'hardtop_convert'}
auto_prices['body_style'] = [body_cats[x] for x in auto_prices['body_style']]
auto_prices['body_style'].value_counts()

#MAKE A BOX PLOT
plot_box(auto_prices, 'body_style')

#%%
def hist_plot(vals, lab):
    sns.distplot(vals)
    plt.title('Histogram of ' + lab)
    plt.xlabel('Value')
    plt.ylabel('Density')

hist_plot(auto_prices['price'], 'prices')

auto_prices['log_price'] = np.log(auto_prices['price'])
hist_plot(auto_prices['log_price'], 'log prices')

#%%
def plot_scatter_shape(auto_prices, cols, shape_col = 'fuel_type', col_y = 'log_price', alpha = 0.2):
      shapes = ['+', 'o', 's', 'x', '^']
      unique_cats = auto_prices[shape_col].unique()
      for col in cols:
          sns.set_style("whitegrid")
          for i, cat in enumerate(unique_cats):
              temp = auto_prices[auto_prices[shape_col] == cat]
              sns.regplot(col, col_y, data=temp, marker = shapes[i], label = cat,
                          scatter_kws={"alpha":alpha}, fit_reg = False, color = 'blue')
          plt.title('Scatter plot of ' + col_y + ' vs. ' + col)
          plt.xlabel(col)
          plt.ylabel(col_y)
          plt.legend()
          plt.show()

num_cols = ['curb_weight', 'engine_size', 'horsepower', 'city_mpg']
plot_scatter_shape(auto_prices, num_cols)

# Let's save the dataframe to a csv file 
# We will use this in the next module so that we don't have to re-do the steps above
# You don't have to run this code as the csv file has been saved under the next module's folder
#auto_prices.to_csv('Auto_Data_Preped.csv', index = False, header = True)



#%%
#Example 2
credit = pd.read_csv('German_Credit.csv', header=None)
credit.head()

credit.columns = ['customer_id', 'checking_account_status', 'loan_duration_mo', 'credit_history', 
                  'purpose', 'loan_amount', 'savings_account_balance', 
                  'time_employed_yrs', 'payment_pcnt_income','gender_status', 
                  'other_signators', 'time_in_residence', 'property', 'age_yrs',
                  'other_credit_outstanding', 'home_ownership', 'number_loans', 
                  'job_category', 'dependents', 'telephone', 'foreign_worker', 
                  'bad_credit']
credit.head()

code_list = [['checking_account_status', 
              {'A11' : '< 0 DM', 
               'A12' : '0 - 200 DM', 
               'A13' : '> 200 DM or salary assignment', 
               'A14' : 'none'}],
            ['credit_history',
            {'A30' : 'no credit - paid', 
             'A31' : 'all loans at bank paid', 
             'A32' : 'current loans paid', 
             'A33' : 'past payment delays', 
             'A34' : 'critical account - other non-bank loans'}],
            ['purpose',
            {'A40' : 'car (new)', 
             'A41' : 'car (used)',
             'A42' : 'furniture/equipment',
             'A43' : 'radio/television', 
             'A44' : 'domestic appliances', 
             'A45' : 'repairs', 
             'A46' : 'education', 
             'A47' : 'vacation',
             'A48' : 'retraining',
             'A49' : 'business', 
             'A410' : 'other' }],
            ['savings_account_balance',
            {'A61' : '< 100 DM', 
             'A62' : '100 - 500 DM', 
             'A63' : '500 - 1000 DM', 
             'A64' : '>= 1000 DM',
             'A65' : 'unknown/none' }],
            ['time_employed_yrs',
            {'A71' : 'unemployed',
             'A72' : '< 1 year', 
             'A73' : '1 - 4 years', 
             'A74' : '4 - 7 years', 
             'A75' : '>= 7 years'}],
            ['gender_status',
            {'A91' : 'male-divorced/separated', 
             'A92' : 'female-divorced/separated/married',
             'A93' : 'male-single', 
             'A94' : 'male-married/widowed', 
             'A95' : 'female-single'}],
            ['other_signators',
            {'A101' : 'none', 
             'A102' : 'co-applicant', 
             'A103' : 'guarantor'}],
            ['property',
            {'A121' : 'real estate',
             'A122' : 'building society savings/life insurance', 
             'A123' : 'car or other',
             'A124' : 'unknown-none' }],
            ['other_credit_outstanding',
            {'A141' : 'bank', 
             'A142' : 'stores', 
             'A143' : 'none'}],
             ['home_ownership',
            {'A151' : 'rent', 
             'A152' : 'own', 
             'A153' : 'for free'}],
            ['job_category',
            {'A171' : 'unemployed-unskilled-non-resident', 
             'A172' : 'unskilled-resident', 
             'A173' : 'skilled',
             'A174' : 'highly skilled'}],
            ['telephone', 
            {'A191' : 'none', 
             'A192' : 'yes'}],
            ['foreign_worker',
            {'A201' : 'yes', 
             'A202' : 'no'}],
            ['bad_credit',
            {2 : 1,
             1 : 0}]]

for col_dic in code_list:
    col = col_dic[0]
    dic = col_dic[1]
    credit[col] = [dic[x] for x in credit[col]]
    
credit.head() 

print(credit.shape)
print(credit.customer_id.unique().shape)

credit.drop_duplicates(subset = 'customer_id', keep = 'first', inplace = True)
print(credit.shape)
print(credit.customer_id.unique().shape)

# Let's save the dataframe to a csv file 
# We will use this in the next module so that we don't have to re-do the steps above
# You don't have to run this code as the csv file has been saved under the next module's folder
#credit.to_csv('German_Credit_Preped.csv', index = False, header = True)

#%%
#FEATURE ENGINEERING
credit[['log_loan_duration_mo', 'log_loan_amount', 'log_age_yrs']] = credit[['loan_duration_mo', 'loan_amount', 'age_yrs']].applymap(math.log)

num_cols = ['log_loan_duration_mo', 'log_loan_amount', 'log_age_yrs',
            'loan_duration_mo', 'loan_amount', 'age_yrs']

for col in num_cols:
    print(col)
    _ = plt.figure(figsize = (10,4))
    sns.violinplot(x = 'bad_credit', y = col, hue = 'bad_credit',
                   data = credit)
    plt.ylabel('value')
    plt.xlabel(col)
    plt.show()