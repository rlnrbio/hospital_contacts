# Networks of Health - Networks of Disease
# Understanding hospital contact patterns using actor-oriented network modelling

## Motivation
Ever since humans have started to systematically and scientifically study diseases, one of the most important questions has been about how diseases are transmitted between individuals. While the SARS-CoV2 pandemic has brought this topic to the forefront of political and scientific discussions in recent years, it has been an intensively researched question for centuries. Not just the SARS-CoV2 pandemic has shown that transmissions in places with high attendance and small distances over a prolonged period of time can be especially consequential. In hospitals, this is exacerbated by the fact that here potential pathogens from infected patients can meet people with a weakened immune system, for example during chemotherapy. According to the Federal Ministry of Health (BMG), between 400,000 and 600,000 patients get infected with hospital diseases in Germany every year with 10,000 - 20,000 of them dying as a result of one of these infections [1](https://www.bundesgesundheitsministerium.de/krankenhaushygiene.html). 
Hospitals are also often a breeding ground for Antibiotic Resistant Germs (ARG) because of a high exposure to patients being treated with antibiotics. Therefore, reducing the number of patients falling ill from hospital infections can also reduce the number of ARG infections drastically. This project looks at modelling the contacts in hospitals based on a complete network approach in order to understand the effects that influence the patterns of contact formation in the health sector. For modelling contacts, two different approaches are chosen here. The first one (Stochastic Actor-Oriented Models (SAOMs)) is based on discrete time steps and the aggregated development of the network in between those moments [2](https://www.sciencedirect.com/science/article/pii/S0378873309000069), the second approach (Dynamic Network Actor Models (DyNAMs)) is using continuous, time stamped data [3](https://doi.org/10.1177/0081175017709295).

## Code
This repository contains the code for both modelling approaches. A more in depth description of the models and their mechanisms can be found in the report also included. 
