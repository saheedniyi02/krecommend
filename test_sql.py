from multiprocessing import connection
from sqlalchemy import create_engine
import pandas as pd
from recommend import KRecommend

engine = create_engine("sqlite:///database.db", echo=True)
connection = engine.connect()

recommender = KRecommend(k=4)
recommender.fit_on_sql_table("Posts", "id",["title","content"], connection)
title="Nasarawa gov tasks traditional rulers on sustained peace, security"
text="""
Nasarawa State Governor, Abdullahi Sule, has charged traditional rulers across the 13 Local Government Areas of the state to ensure that they sustain the peace being enjoyed in their respective communities and report anyone found causing trouble to security agencies.

Speaking at Agyaragu, Obi Local Government Area, on Monday, during the ’25 years on the throne celebration’ of the Zhe Migili, His Royal Majesty, Dr. Ayuba Agwadu, Sule said the campaign on peace and security became important because no society could be developed in an atmosphere of rancour.

He said, “Nasarawa State is one of the most peaceful states in the country because of the efforts and support from the traditional rulers.

“I want to also appeal to all residents of the state to always be vigilant and report suspicious movements in their environments to the security agencies because security is everyone’s business.”



Governor Abdullahi Sule, who was represented by his deputy, Emmanuel Akabe, noted that the contributions of the royal father in peace building had complemented the efforts of the state government in maintaining peace in the area.

He warned against any form of violence during the primary elections of political parties and the 2023 general election, saying that troublemakers would be arrested and made to face the wrath of the law.

“As the 2023 elections draw near, I want to caution all residents against disrupting the existing peace in the state. Instead of causing problems, we should work together to ensure that credible leaders are elected into political offices to ensure steady growth and development of our dear state,” Sule added.


The celebrant, Zhe Migili of the Koro tribe in the state, Ayuba Agwadu, expressed gratitude to God for making it possible for him to attain the milestone of 25 years on the throne.

He also commended his subjects for their support and cooperation in steering the throne in the past 25 years.

While highlighting his personal experiences, developments and challenges over the years, the Zhe Migili appealed to the federal and state governments to give political appointments to his people to enable them bring the needed social amenities to Koro communities.

Our correspondent reports that members of the National and State Assemblies, political appointees and traditional rulers from Nasarawa and neighbouring states, attended the ceremony.

"""
recommended=recommender.predict_on_sql_table([title,text])
print(recommended)