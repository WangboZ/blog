---
layout: post
title: "How to Authenticate to Azure Machine Learning Workspace in a Defined Tenant"
tagline: 
author: "Wangbo Zheng"
---

[Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/) is a cloud-based environment where you can use to train, deploy, automate, manage, and track ML models.  Whether you prefer to write Python or R code or work with no-code/low-code options in the studio, you can build, train, and track machine learning and deep-learning models in an Azure Machine Learning Workspace. Data scientists and developers can access the assets of an Azure ML Workspace not only with the web portal ([Azure Machine Learning studio](https://docs.microsoft.com/en-us/azure/machine-learning/overview-what-is-machine-learning-studio)) but also with [Azure ML Python SDK](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py). 

In this article, you will learn how to authenticate to an Azure Machine Learning Workspace with Python, whether it is in your default tenant or not. There are several different methods to access an ML Workspace in a subscription in a defined (not default) tenant. Here we will discuss it with two scenarios: local computer and compute instance in ML Workspace. 

## Access to ML Workspace in a Default Tenant

Interactive authentication is the default mode when using Azure ML SDK. When you connect to your workspace using ```workspace.from_config```, you will get an interactive login dialog. The *config.json* file can be downloaded in ML Workspace in [Azure Portal](https://portal.azure.com). It can also be edited manually with the following template

```json
{
    "subscription_id": "my-subscription-id",
    "resource_group": "my-ml-rg",
    "workspace_name": "my-ml-workspace"
}
```

```python
from azureml.core import Workspace
ws = Workspace.from_config()
```

Also, you can use the following code with the subscription ID, resource group, and workspace name as input arguments.

```python
ws = Workspace(subscription_id="my-subscription-id",
               resource_group="my-ml-rg",
               workspace_name="my-ml-workspace")
```




## Access to ML Workspace in a Defined Tenant on a Local Computer

With the above-mentioned method, if you get an error message like: ```All the subscriptions that you have access to = []```
In such a case, the ML Workspace you want to access might not in your default tenant. You may have to name the tenant ID of the Azure Active Directory you're using. You specify the tenant by explicitly instantiating ```InteractiveLoginAuthentication``` with Tenant ID as an argument. The Tenant ID can be found, for example, from [Azure Portal](https://portal.azure.com) under Azure Active Directory, Properties as Directory ID.

```python
from azureml.core.authentication import InteractiveLoginAuthentication

interactive_auth = InteractiveLoginAuthentication(tenant_id="my-tenant-id")

ws = Workspace(subscription_id="my-subscription-id",
               resource_group="my-ml-rg",
               workspace_name="my-ml-workspace",
               auth=interactive_auth)
```

## Access to ML Workspace in a Defined Tenant on Compute Instance

If you want to access the ML Workspace not in your default tenant on a compute instance (Notebooks) in ML Workspace, the ```InteractiveLoginAuthentication```method also does not always work. Through personal experience, the following methods with ```AzureCliAuthentication``` can solve it. 

First type following code in a terminal (opened with your Jupyter Notebook) or in a Jupyter Notebook cell with``` ! ```magic. If you don't know which tenant you can access, you can also type ```az login``` only, you can then see all the information of your tenants after you logged in. Select the right tenant and type the following code again.

```bash
az login --tenant MY_TENANT
```

After you logged in with [Azure CLI](https://docs.microsoft.com/de-de/cli/azure/authenticate-azure-cli), you can run the following code in your notebook.

```python
from azureml.core.authentication import AzureCliAuthentication

cli_auth = AzureCliAuthentication()

ws = Workspace(subscription_id="my-subscription-id",
               resource_group="my-ml-rg",
               workspace_name="my-ml-workspace",
               auth=cli_auth)

print("Found workspace {} at location {}".format(ws.name, ws.location))
```

This method also works on a local computer if you have installed [azure-cli package](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).





Hope this article can help you. There are other ways of authentication, such as Managed Service Identity (MSI) authentication
service principal authentication, token authentication. You can find more detail in this [notebook](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb).

### References

1. [MachineLearningNotebooks/authentication-in-azureml.ipynb at master · Azure/MachineLearningNotebooks · GitHub](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb)

2. [Sign in with the Azure CLI](https://docs.microsoft.com/de-de/cli/azure/authenticate-azure-cli)

3. [What is the antonym for "default"? - English Language Learners Stack Exchange](https://ell.stackexchange.com/questions/22051/what-is-the-antonym-for-default)

   

   

