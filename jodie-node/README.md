## JODIE for node classification
#### Authors: [Srijan Kumar](http://cs.stanford.edu/~srijan), [Xikun Zhang](), [Jure Leskovec](https://cs.stanford.edu/people/jure/)
<!--#### [Project website with links to the datasets](http://snap.stanford.edu/jodie/)-->
#### [Link to the paper](https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019.pdf)
#### [Link to the slides](https://cs.stanford.edu/~srijan/pubs/jodie-kdd2019-slides.pdf)
#### [Brief video explanation](https://www.youtube.com/watch?v=ItBmU8681j0)

### Introduction
Temporal networks are ubiquitous in e-commerce (users clicking, purchasing, saving items), social networks (users talking with one another and interacting with content), finance (transactions between users and merchants), and education (students taking courses). In all domains, the entities (users, items, content) can be represented as nodes and their interaction as edges. 

**JODIE** is a representation learning framework for all nodes in temporal networks. Given a sequence of node actions, JODIE learns a dynamic embedding trajectory for every node (as opposed to a static embedding). These trajectories are useful for downstream machine learning tasks, such as link prediction, node classification, and clustering. JODIE is fast and makes accurate predictions about future interactions and anomaly detection.

In this paper, JODIE has been used for two broad category of tasks:
1. **Temporal Link Prediction**: Which two nodes will interact next? Example applications are recommender systems and modeling network evolution.
2. **Temporal Node Classification**: When does the state of an node change from normal to abnormal? Example applications are anomaly detection, ban prediction, dropout and churn prediction, and fraud and account compromise.
