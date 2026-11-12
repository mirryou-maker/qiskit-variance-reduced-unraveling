import re
s=open('quantum_journal.tex',encoding='utf8').read()
bib=set(re.findall(r'bibitem\{([^}]*)\}',s))
cit=set(k.strip() for c in re.findall(r'cite\{([^}]*)\}',s) for k in c.split(','))
print("uncited:",sorted(bib-cit)); print("missing:",sorted(cit-bib)); print(len(bib),len(cit))
