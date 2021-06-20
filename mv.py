

def apply_rule(row):

	c = Counter(row)
	max_count = c.most_common()[0][1]

	polemic_classes = []
	for key in c:
		if c[key] == max_count:
			polemic_classes.append(key)

	if len(polemic_classes) == 1:
		return polemic_classes[0]
	else:
		if 0 in polemic_classes and 2 in polemic_classes:
			return 1
		else:
			return 2

vote = df.astype(int).apply(lambda row: apply_rule(row), axis=1)