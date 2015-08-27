# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 11:08:17 2015

@author: fraidylevilev
"""

# My access token: c2f218e5c3a9af9a3e0389d3e539e197f19f650e
# My athlete id: 9753705

from stravalib.client import Client
# import the strava library

client = Client()

# Input your access token below!
client.access_token = 'c2f218e5c3a9af9a3e0389d3e539e197f19f650e'
#this is me
athlete = client.get_athlete(9753705)
print("For {id}, I now have an access token {token}".format(id = athlete, token = client.access_token))

import requests
import pandas as pd
import numpy as np
import time
import sys

#calling strava api
base_url = 'https://www.strava.com/api'
#using api to get efforts on a segment
segment_url = base_url + '/v3/segments/{0}/all_efforts'
extra_headers = {'Authorization' : 'Bearer {0}'.format(access_token)}
per_page = 200

#gets first 30 attempts for a segment
r = requests.get(segment_url.format(9983525), headers=extra_headers)
results = r.json()

# input:  segment 
# output:  list of athletes
def get_people(segment_id, pages = 1):
    #access_token = 'c2f218e5c3a9af9a3e0389d3e539e197f19f650e'
    #extra_headers = {'Authorization' : 'Bearer {0}'.format(access_token)}
    request_to_strava = requests.get('https://www.strava.com/api/v3/segments/{0}'.format(segment_id), headers=extra_headers).json()
    effort_count = request_to_strava['effort_count']
    print effort_count
    segment_url = 'https://www.strava.com/api/v3/segments/{0}/all_efforts'.format(segment_id)
    print segment_url
    params = {}
    params['start_date_local'] = '2015-07-01T00:00:00Z'
    params['end_date_local'] = '2016-01-01T23:59:59Z'
    params['per_page'] = 200
    
    all_efforts = []
    
    for number in range(1,pages + 1):
        print number
        params['page'] = number
        segment_request = requests.get(segment_url, params = params, headers=extra_headers).json()
        all_efforts += segment_request

    new_efforts = []
    
    for effort in all_efforts:
        new_efforts.append( {
        'athlete_id': effort['athlete']['id'],
        'segment_id':segment_id,
        'average_watts': effort.get('average_watts', -1),
        'elapsed_time': effort['elapsed_time'],
        'moving_time': effort['moving_time'],
        'average_grade': effort['segment']['average_grade'],
        'distance': effort['segment']['distance'],
        'elevation_range': effort['segment']['elevation_high'] - effort['segment']['elevation_low']
        })
    return pd.DataFrame(new_efforts)
    
    
people = get_people(652851, pages = 1)

people

len(people)

def get_segment_details(a_segment):
    segment_results = []
    r = requests.get(segment_url.format(segment), headers=extra_headers)
    results = r.json()
    segment_results = {'segment_id': segment,
                        'seg_name': results['name'],
                       'seg_city': results['city'],
                       'avg_grade': results['average_grade'],
                       'distance': results['distance'],
                       'elev_gain': results['total_elevation_gain']}
    return segment_results

#df_seg = pd.DataFrame(segment_results)

segments = [229781, 4313, 241885, 2371095, 2451142, 612695, 2324148, 611787, 652196, 688554]

#MAKE FINAL EFFORTS TABLE
efforts = []
for segment in segments:
    efforts_for_segment = get_people(segment, pages = 3)
    efforts.append(efforts_for_segment)

final_efforts = pd.concat(efforts)

#MAKE FINAL SEGMENTS TABLE
segment_details = []
for segment in segments:
    details = get_segment_details(segment)
    segment_details.append(details)
    print segment_details

final_segments = pd.DataFrame(segment_details)

final = pd.merge(final_efforts,final_segments,how='left',left_on='segment_id',right_on='segment_id')
