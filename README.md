[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hibernator11/notebook-emerging-topics-corpora/master)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4636823.svg)](https://doi.org/10.5281/zenodo.4636823)



# Introduction

The exploration of trends and temporal variations in the digital collections is possible by means of computational analysis. For example, users may be interested in the early detection of emerging topics (novelties in the content). In this sense, emerging subjects in scientific papers and journals can help funding agencies to identify potential and new areas of research.

This project includes a collection of Jupyter Notebooks to work with text corpora provided by relevant GLAM institutions to identify emerging topics. Each notebook describes how the metadata is retrieved and processed in order to obtain as a result emergent topics. A preliminar Jupyter Notebook explain how the method works step by step.


### Introduction to text analysis
This [notebook](https://nbviewer.jupyter.org/github/hibernator11/notebook-emerging-topics-corpora/blob/master/introduction-to-text-analysis.ipynb) introduces how to analyse text to identify topic trends in text corpora.


### Biblioteca Virtual Miguel de Cervantes LOD & the journal Doxa
This [notebook](https://nbviewer.jupyter.org/github/hibernator11/notebook-emerging-topics-corpora/blob/master/doxa.ipynb) uses the Linked Open Data repository of the [Biblioteca Virtual Miguel de Cervantes](http://www.cervantesvirtual.com).

This example is based on the journal *Doxa. Cuadernos de Filosofía del Derecho* that is a periodical publication issued every year since 1984 to promote the interchange between philosophers of law from Latin America and Latin Europe. The information regarding this publication has been published as LOD in the BVMC repository, including metadata and text, and is accessible by means of the SPARQL endpoint.

<img src="images/journal-relationships.png">

The results obtained as a result were assessed by the digital curators at the digital library.


### UK Doctoral Thesis Metadata from EThOS

This [notebook](https://nbviewer.jupyter.org/github/hibernator11/notebook-emerging-topics-corpora/blob/master/ethos.ipynb) is based on the [UK Doctoral Thesis Metadata from EThOS](https://doi.org/10.23636/1344) that comprises the bibliographic metadata for all UK doctoral theses listed in EThOS, the UK's national thesis service.

This additional [notebook](https://nbviewer.jupyter.org/github/hibernator11/notebook-emerging-topics-corpora/blob/master/ethos-subjects.ipynb) analyses the most common subjects for a particular term that appears in the abstracts.


### dblp computer science bibliography

This [notebook](https://nbviewer.jupyter.org/github/hibernator11/notebook-emerging-topics-corpora/blob/master/dblp.ipynb) is based on the [dblp computer science bibliography](https://dblp.uni-trier.de/xml/) that comprises a long list of publication records in XML format. Two periods (2000-2010 and 2010-2021) are assessed in order to have an overview of how the method is applied.


### Library of Congress and Chronicling America digitized newspaper collection

This [notebook](https://nbviewer.jupyter.org/github/hibernator11/notebook-emerging-topics-corpora/blob/master/chronicling-america-loc.ipynb) is based on the [Bourbon News](https://chroniclingamerica.loc.gov/lccn/sn86069873/) which was initially a simple, eight-page publication, offering state, national, and international news, as well as serials, anecdotes, and topical items. The county's economic interests and its cultural passions are reflected in the publication including the horse industry and the tobacco business. The paper continued publishing until 1941.


<img src="images/graph-loc.png">

## References

- Mahey, M., Al-Abdulla, A., Ames, S., Bray, P., Candela, G., Chambers, S., Derven, C., Dobreva-McPherson, M., Gasser, K., Karner, S., Kokegei, K., Laursen, D., Potter, A., Straube, A., Wagner, S-C. and Wilms, L. with forewords by: Al-Emadi, T. A., Broady-Preston, J., Landry, P. and Papaioannou, G. (2019) Open a GLAM Lab. Digital Cultural Heritage Innovation Labs, Book Sprint, Doha, Qatar, 23-27 September 2019.

- Padilla, Thomas, Allen, Laurie, Frost, Hannah, Potvin, Sarah, Russey Roke, Elizabeth, & Varner, Stewart. (2019, May 22). Final Report --- Always Already Computational: Collections as Data (Version 1). Zenodo. http://doi.org/10.5281/zenodo.3152935

- Candela, G. et al. (2020) ‘Reusing digital collections from GLAM institutions’, Journal of Information Science. doi: 10.1177/0165551520950246.
