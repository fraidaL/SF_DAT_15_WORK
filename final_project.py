#Strava Final Project
#By: Fraida Levilev

#Strava is an app that tracks cycling (and running, and other things, but I am focusing 
#on cycling) data. Within a ride are various segments and cyclists (who have a slight 
#obsession with strava) will compete for leaderboard positions on popular segments. If 
#you have the fastest time on a segment, you are crowned the KOM/QOM (King/Queen of 
#the Mountain). 

#What I want to do is look at a segment (the gradient and distance) and predict how fast 
#someone will be given their performance (time, speed…) on similar segments. So far I 
#have this function which when you input a segment will output the list of athletes that 
#have completed said segment. This one is only set to output the first 3 pages so that I 
#don’t use all my API calls while I am testing it. 



def get_people(segment_id, pages = 1):
    access_token = 'c2f218e5c3a9af9a3e0389d3e539e197f19f650e'
    extra_headers = {'Authorization' : 'Bearer {0}'.format(access_token)}
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
        segment_request = requests.get(segment_url, params = params, headers = extra_headers).json()
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
    return new_efforts 
    
    
people = get_people(652851, pages = 3)

people