---
layout: post
title: "How to prepare for DP-100 the Azure Data Scientist Associate Test"
tagline: 
author: "Wangbo Zheng"
---

An Azure Data Scientist Associate Badge looks cool. If you also want to get yours, you need to pass the DP-100 exam: Designing and Implementing a Data Science Solution on Azure. In this article, I will tell you what the content of the DP-100 exam is, how did I prepare for it. I hope after you finish reading this article, it will help you to pass the DP-100 exam.

![badge](https://raw.githubusercontent.com/WangboZ/blog/master/image/azure-data-scientist-associate-600x600.png){:height="50%" width="50%"}

I took the exam end of August. According to the official exam website, this exam was updated on December 8. The exam skills outline does not change so much. Some terminology is modified slightly, which happens all the time in Azure. 



## What are the Azure Data Scientist Associate Certification and DP-100 exam?

The definition of the Azure Data Scientist is that he/she applies their knowledge of data science and machine learning to implement and run machine learning workloads on Azure; in particular, using Azure Machine Learning Service. DP-100 exam is the exam you need to pass to earn the Azure Data Scientist Associate Certification.



The exam can be taken at home or in an exam center. Due to the epidemic, most people choose to take the exam at home, all you need is a suitable computer and a room that meets the requirements. The other is that you have to be on time. You have **180 minutes** for the exam. There are some surveys before the exam, and you also have time to make suggestions for each exam question after the exam. In the exam, you will get **55 questions**, such as: 

- Multiple choice
- True/False (Yes/No)
- Arrange in the correct order
- Drop-down/ Dialog box

There are some questions you have a drop-down menu to select the correct **code snippet or configuration setting**.  Also, there are some series of questions that present the same scenario. This question you can **not review** after you answer them. 



The skills measured in the exam consists of 4 parts:

- Set up an Azure Machine Learning workspace (30-35%)
- Run experiments and train models (25-30%)
- Optimize and manage models (20-25%)
- Deploy and consume models (20-25%)



I learned from my colleagues that this exam used to contain many questions about data science and statistic. The recently updated version includes more questions about the Azure Machine Learning workspace. It is difficult to pass the exam without Azure experience and adequate preparation.



## How to prepare for it?

I work as a data scientist for more than two years. Benefit from the partnership between our company and Microsoft, I have already gained many experiences with Azure, especially Azure Machine Learning related services. As mentioned before, not only data science basic knowledge is necessary for the exam, but machine learning related Azure skills are also more important. It's not a problem if you don't have an Azure account yet. You can create your [Azure free account ]((https://azure.microsoft.com/en-us/free/)) with 170€/200$ credits for 30 days. It is enough for you to practice with the Azure Machine Learning service even also other Azure products. 



Of course, you will gain all the skills needed to become certified, if you have the opportunity to take part in paid instructor-led courses. But there is also plenty of online available free learning stuff that can support you pass the exam. If you can remember all the detail in the documentation of [Azure Machine Learning services](https://docs.microsoft.com/en-us/azure/machine-learning/), you will have more than enough power to conquer the exam. 



The most helpful resources come from [Microsoft Learn](https://docs.microsoft.com/en-us/learn/). You will get all the free courses you needed by selecting data scientist as your role. The following learning paths almost all exam requirements. 

![MicrosoftLearn](https://raw.githubusercontent.com/WangboZ/blog/master/image/microsoft-learn.png)

There are always exercises in each Microsoft Learn course. For the drag and drop style **Azure Machine Learning designer** you can get step by step guide with pictures. There are also code snippets with explanations for the **Python SDK**. You can find the corresponding notebooks in this [Github repository]([GitHub - MicrosoftLearning/mslearn-dp100: Lab files for Azure Machine Learning exercises](https://github.com/MicrosoftLearning/mslearn-dp100)).Make sure that you go through all the courses with the hands-on lab at least once. Because there will be many detailed questions in the exam. Sometimes you need to fill a blank of code (with option) or find the suitable module. 



The only disadvantage of Microsoft Learn is that most courses are only text. If you are easily bored while reading, you can also find the [video tutorials in Pluralsight](https://www.pluralsight.com/paths/microsoft-azure-data-scientist-dp-100) for the DP-100 exam. There are many courses related to Azure Machine Learning service, make sure they match the exam syllabus before watching them. 



DP-100 is not the first Azure exam I passed, like many other exams, Azure exams will also have a lot of questions about concepts that people tend to confuse. Sometimes, there is not only one correct solution for the scenario-based question series. So when you learn for the exam, pay some attention to **comparisons between concepts and solutions**. For example, you need to know which **evaluation metrics** are suitable for classification or regression problem. Also, the **compute target** option to host a machine learning model in Azure is a frequent problem. Because the compute target you choose will affect the cost and availability of the deployed endpoint. The next example is about the **explainer**. There are multiple types of explainer, you need to know which one use SHAP algorithms and which one doesn't support local feature importance explanations. Write a summary or make some tables will help you remember these details.



## Why get certified?

After you prepared/passed the DP-100 exam, you will have a clear view of the whole industry data science/machine learning project pipeline in the cloud environment, from training model to optimizing model, from deploying to monitoring the final product. 



```
> Upon earning a certification, 23% of Microsoft certified technologists reported receiving up to a 20% salary increase. What’s more, certified employees are often entrusted with supervising their peers—putting them on the fast track for a promotion. —2017 Pearson VUE Value of Certification white paper.
```

