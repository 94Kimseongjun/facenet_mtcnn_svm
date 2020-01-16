name="test"
version="3.6.2"

text = open('Dockerfile','w')
data="FROM continuumio/miniconda3\nMAINTAINER Seongjun Kim<s.kim@xiilab.com>\nRUN conda update -n base conda\nRUN conda create -n " + name + " python=" + version
print(data)
text.write(data)


