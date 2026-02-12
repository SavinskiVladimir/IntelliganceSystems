import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('data/medicines.csv').set_index('TransactionID')
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
print(rules[['antecedents', 'consequents', 'support', 'confidence']])
