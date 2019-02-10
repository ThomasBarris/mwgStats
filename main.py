#!/usr/bin/env python3

import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from scipy import stats
# requires xlrd for reading Excel

# 'member' incl 12 months grace period, otherwise strict i.e. paid, eligible for voting
logic = 'member'

# input file from CiviCRM
input_file_location = r"C:\....\Report_20190209-659.csv"

# member data for time before CiviCRM
history_file_location = r"C:\....\pre_civic.xlsx"

# where you want to have your charts saved
chart_location = r"C:\....\Member Stats\\"

# numbers of mappers per country from Pascal Neis
# https://tools.neis-one.org/tmp/20181118_AvgOSMContributors.txt
mappers_file = r"C:\.....\20181118_AvgOSMContributors.txt"

# number of month to read from CiviCRM dump. 37 = January 2019, increase as appropriate
data_range = 37

# charts will be displayed on screen, set to True if you don't want that
headless = False

# matplotlib figsize for standard charts i.e. except bar chart with OSMF members and mappers per country
figsize_x = 8  # default value 8
figsize_y = 6  # default value 6


def str2date(s):
    """converts a string from CiviCRM to a date"""
    d = None
    try:
        d = datetime.datetime.strptime(s, "%Y-%m-%d")
    finally:
        return d


def make_neat(jumble):
    """read thre CiviCRM csv file
    returns a unique list of members with
    [0] Membership Type,    [1] = Start Date,   [2] = End Date,     [3] = Join Date
    [4] = E-Mail,           [5] = name,         [6] = country
    """
    return_list = list()
    for item in jumble:
        # if chancelled, a valid entry exists too with different dates, so it would be counted twice otherwise
        if item[5] != 'Chancelled':
            return_list.append(
                [item[1], str2date(item[2]), str2date(item[3]), str2date(item[4]), item[6], item[0], item[7]])

    # make list unique
    return_list = [list(x) for x in set(tuple(x) for x in return_list)]

    return return_list


def by_months(l_members, l_data_range):
    """generates a list with numbers for each member category for each month
    returns list with
    [0] date   [1] No of normal Members    [2] No of Associate Members,
    [3] No of Fee Waiver Member            [4] No of Corporate Member
    """
    return_list = list()
    for i in range(0, l_data_range):
        finished_members = set([])  # store members we have already processed
        if logic == 'member':
            d1 = datetime.datetime(2016 + int(i / 12), i % 12 + 1, 1)
            d2 = datetime.datetime(2015 + int(i / 12), i % 12 + 1, 1)
        else:
            d1 = datetime.datetime(2016 + int(i / 12), i % 12 + 1, 1)
            d2 = datetime.datetime(2016 + int(i / 12), i % 12 + 1, 1)
        s = dict()
        for m in l_members:
            tester = m[4] + m[5]  # kind of hash to remember already processed members
            if m[1] and m[1] < d1 and d2 < m[2] and tester not in finished_members:
                s[m[0]] = s.get(m[0], 0) + 1
                finished_members.add(tester)
        return_list.append([datetime.datetime(2016 + int(i / 12), i % 12 + 1, 1), s.get('Normal Member', 0),
                            s.get('Associate Member', 0), s.get('Fee-waiver Member', 0), s.get('Corporate Member', 0)])
    return return_list


def by_country(l_members, l_data_range):
    """returns a list with number of members without Corporate Members for each country
    returned list is descending sorted by number of members
    returns list with
    [0] name [1] number of members ex corp"""
    countries = dict()

    end_period = l_data_range - 1

    d1 = datetime.datetime(2016 + int(end_period / 12), end_period % 12 + 1, 1)
    d2 = datetime.datetime(2015 + int(end_period / 12), end_period % 12 + 1, 1)

    finished_members = set([])  # store members we have already processed

    for m in l_members:
        tester = m[4] + m[5]
        if m[0] != "Corporate Member" and m[1] and m[1] <= d1 and d2 < m[2] and tester not in finished_members:
            countries[m[6]] = countries.get(m[6], 0) + 1
            finished_members.add(tester)

    sorted_country_list = list()
    for item in sorted(countries, key=countries.get, reverse=True):
        sorted_country_list.append([item, countries[item]])

    return sorted_country_list


def print_member_history_chart(l_total_history_df):
    """generates a
    - line chart with the different member types by months
    - stacked area chart with the different member types by months
    saves charts as *.png and displays the charts if headless != True"""

    # print the line chart
    # select a nice stylesheet
    # https://matplotlib.org/examples/style_sheets/style_sheets_reference.html
    matplotlib.style.use('bmh')
    ax = l_total_history_df[['normalMember', 'associateMember', 'feeWaiverMember', 'corporateMember']].plot(
        figsize=(figsize_x, figsize_y))
    ax.legend(('Normal Members', 'Associate Members', 'Fee-waiver Members', 'Corporate Members'), loc="upper left")

    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.ylim([0, None])
    plt.title("OSMF Membership Statistics", fontweight='bold', fontsize=14)
    filename = 'history.png'
    filename = chart_location + filename
    plt.savefig(filename)
    if not headless:
        plt.show()

    # print stacked area chart
    ax = l_total_history_df[['feeWaiverMember', 'corporateMember', 'associateMember', 'normalMember']].plot.area(
        figsize=(figsize_x, figsize_y))
    ax.legend(('Fee-waiver Members', 'Corporate Members', 'Associate Members', 'Normal Members'), loc="upper left")
    ax.set_ylabel('Numbers stacked')
    ax.set_xlabel('')
    plt.title("OSMF Membership Statistics", fontweight='bold', fontsize=14)
    filename = 'history2.png'
    filename = chart_location + filename
    plt.savefig(filename)
    if not headless:
        plt.show()


def print_by_continent(l_history_by_continent_df):
    """generates a
    - line chart with number of members by continent
    - stacked area chart with number of members by continent
    saves charts as *.png and displays the charts if headless != True

    needs input dataframe with OSMF members in columns
     [0] date [1] africa [2] asia [3] europe [4] north america [5] oceanian [6] south america [7] other"""

    # print the line chart
    # select a nice stylesheet
    # https://matplotlib.org/examples/style_sheets/style_sheets_reference.html
    # columns = 'date', 'Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania', 'unknown'
    matplotlib.style.use('seaborn')
    ax = l_history_by_continent_df[['unknown', 'Oceania', 'South America', 'Africa', 'Asia', 'North America', 'Europe']]\
        .plot(figsize=(figsize_x, figsize_y))
    ax.legend(('unknown', 'Oceania', 'South America', 'Africa', 'Asia', 'North America', 'Europe'), loc="upper left")
    ax.set_ylabel('')
    ax.set_xlabel('')
    plt.ylim([0, None])
    plt.title("OSMF Membership Statistics", fontweight='bold', fontsize=14)
    plt.figtext(0.01, 0.01, 'ex Corporate Members', horizontalalignment='left', fontsize=8)
    filename = 'historyContinent.png'
    filename = chart_location + filename
    plt.savefig(filename)
    if not headless:
        plt.show()

    # print stacked area chart
    ax = l_history_by_continent_df[['unknown', 'Oceania', 'South America', 'Africa', 'Asia', 'North America', 'Europe']]\
        .plot.area(figsize=(figsize_x, figsize_y))
    ax.legend(('unknown', 'Oceania', 'South America', 'Africa', 'Asia', 'North America', 'Europe'), loc="upper left")
    ax.set_ylabel('Numbers stacked')
    ax.set_xlabel('')
    plt.title("OSMF Membership Statistics", fontweight='bold', fontsize=14)
    plt.figtext(0.01, 0.01, 'ex Corporate Members', horizontalalignment='left', fontsize=8)
    filename = 'historyContinent2.png'
    filename = chart_location + filename
    plt.savefig(filename)
    if not headless:
        plt.show()


def pie_print_by_country(l_members_by_country, l_data_range):
    """generates a
     - donut chart with number of members by country
     saves chart as *.png and displays the chart if headless != True"""

    # minimum percent of a country to be labeled in the chart
    label_min = 1.0 / 100

    day = 1
    months = (l_data_range - 1) % 12 + 1
    year = 2016 + int(data_range - 1) / 12
    date = str(int(year)) + '/' + str(months) + '/' + str(day)

    labels = list()
    sizes = list()
    total = 0
    for l_line in l_members_by_country:
        total = total + l_line[1]

    for l_line in l_members_by_country:
        if float(l_line[1]) / total > label_min:
            labels.append((l_line[0] + ' (' + '{:.1%}'.format(float(l_line[1]) / total) + ')'))
        else:
            labels.append(' ')
        sizes.append(l_line[1])

    fig, ax = plt.subplots(gridspec_kw=dict(bottom=0.05), figsize=(figsize_x, figsize_y))
    plt.title("OSMF Membership Statistics", fontweight='bold', fontsize=14 )

    ax.pie(sizes, labels=labels, startangle=90, textprops={'fontsize': 12})

    plt.figtext(0.01, 0.01, ('as of ' + date + ' ex Corporate Members'), horizontalalignment='left', fontsize=8)

    # add some white spaces to avoid overlaps and broken labels
    plt.subplots_adjust(bottom=0.05, right=0.87)

    # draw with circle that makes a pie chart to a donut chart
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax.axis('equal')

    plt.tight_layout()
    filename = 'byCountry.png'
    filename = chart_location + filename
    plt.savefig(filename)
    if not headless:
        plt.show()


def bar_print_by_country(l_members_by_country, l_data_range, l_mappers_file):
    """generates a
     - bar chart
        i) with number of OSMF members by country
        ii) with number of mappers per country, loaded from Pascal Neis' input file
     saves chart as *.png and displays the chart if headless != True
     special request from user stereo i.e. Guillaume Rischard
     Must be scaled huge to be readable
     """

    # minimum percent of a country to be labeled in the chart
    label_min = 0.3 / 100

    day = 1
    months = (l_data_range - 1) % 12 + 1
    year = 2016 + int(l_data_range - 1) / 12
    date = str(int(year)) + '/' + str(months) + '/' + str(day)

    labels = list()
    osmf_sizes = list()
    mapper_sizes = list()
    mappers = dict()

    total = 0
    for l_line in l_members_by_country:
        total = total + l_line[1]

    with open(l_mappers_file) as f:
        content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
    for l_line in content:
        country = l_line.split("\t")
        mappers[country[0]] = country[1]

    for l_line in l_members_by_country:
        if float(l_line[1]) / total >= label_min:
            labels.append((l_line[0] + ' (' + '{:.1%}'.format(float(l_line[1]) / total) + ')'))
        else:
            labels.append(' ')
        osmf_sizes.append(l_line[1])


        if l_line[0] == "Iran, Islamic Republic of":
            mapper_sizes.append(mappers.get('Iran', 0))
        elif l_line[0] == "Russian Federation":
            mapper_sizes.append(mappers.get('Russia', 0))
        elif l_line[0] == "Côte d'Ivoire":
            mapper_sizes.append(mappers.get('Ivory Coast', 0))
        elif l_line[0] == "Congo, The Democratic Republic of":
            mapper_sizes.append(mappers.get('Congo - Kinshasa', 0))
        elif l_line[0] == "Myanmar":
            mapper_sizes.append(mappers.get('Myanmar(Burma)', 0))
        elif mappers.get(l_line[0], 0) == 0:
            print(l_line[0], ' not found')
            mapper_sizes.append(mappers.get(l_line[0], 0))
        else:
            mapper_sizes.append(mappers.get(l_line[0], 0))

    l_mapper_members_df = pd.DataFrame(
        {'Country': labels,
         'OSMF Members': osmf_sizes,
         'OSM Mappers': mapper_sizes
         })

    print(l_mapper_members_df)
    l_mapper_members_df['OSM Mappers'] = l_mapper_members_df['OSM Mappers'].astype(np.int64)

    y_pos = np.arange(len(labels))

    fig, ax = plt.subplots()

    # Create horizontal bars
    l_mapper_members_df.plot(kind='barh', legend=True, fontsize=8, figsize=(28, 12), ax=ax)

    plt.title("OSMF Membership Statistics", fontweight='bold', fontsize=14)

    plt.figtext(0.01, 0.01, ('OSMF as of ' + date + '- Mapper as of Nov 2018 - ex Corporate Members' + ' - Countries with less than {:.1%}'.format(float(label_min)) + ' left blank intentionally'), horizontalalignment='left', fontsize=8)

    # plt.subplots_adjust(right=0.97, left=0.08, bottom=0.05)

    # Create names on the y-axis
    plt.yticks(y_pos, labels)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.05)
    filename = 'byCountry1.png'
    filename = chart_location + filename
    plt.savefig(filename)
    if not headless:
        plt.show()

    return l_mapper_members_df


def mapper_member_regression(l_mapper_members_df):
    """"prints a chart with a linear regression of OSMF members dependent on mappers
    with regression function and coefficient of determination r-squared"""

    # get coeffs of linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(l_mapper_members_df['OSM Mappers'],
                                                                   l_mapper_members_df['OSMF Members'])
    r_squared = r_value * r_value
    label = "y={0:.1f}x+{1:.1f}".format(slope, intercept)
    label = label + " / rsquared:{:10.2f}".format(r_squared)
    ax = sns.regplot(x='OSM Mappers', y='OSMF Members', data=l_mapper_members_df,
                     line_kws={'label': label})
    # plot legend
    ax.legend()
    plt.title("OSMF Membership Regression OSMF Members Based on Mappers", fontweight='bold', fontsize=14)
    filename = 'memberRegression.png'
    filename = chart_location + filename
    plt.savefig(filename)
    if not headless:
        plt.show()


def continent_history(l_members, l_data_range):
    """returns a list with number of members per continent for each months
     returned list is descending sorted by number of members

     input list [0] Membership Type,    [1] = Start Date,   [2] = End Date,     [3] = Join Date
                [4] = E-Mail,           [5] = name,         [6] = country

     returns list with OSMF members in
     [0] date [1] africa [2] asia [3] europe [4] north america [5] oceanian [6] south america [7] other"""
    return_list = list()

    africa = {"Algeria", "Angola", "Benin", "Botswana", "Burkina", "Burkina Faso", "Burundi", "Cameroon", "Cape Verde",
              "Central African Republic", "Chad", "Comoros", "Congo", "Congo, The Democratic Republic of the",
              "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Ethiopia", "Gabon", "Gambia", "Ghana", "Guinea",
              "Guinea-Bissau", "Ivory Coast", "Côte d'Ivoire", "Kenya", "Lesotho", "Liberia", "Libya", "Madagascar",
              "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", "Namibia", "Niger", "Nigeria",
              "Rwanda", "Sao Tome and Principe", "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa",
              "South Sudan", "Sudan", "Swaziland", "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"}
    asia = {"Afghanistan", "Bahrain", "Bangladesh", "Bhutan", "Brunei", "Myanmar", "Cambodia", "China", "Hong Kong",
            "East Timor", "India", "Indonesia", "Iran", "Iraq", "Israel", "Japan", "Jordan", "Kazakhstan",
            "Korea, Republic of", "Kuwait", "Kyrgyzstan", "Laos", "Lebanon", "Malaysia", "Maldives", "Mongolia",
            "Nepal", "Oman", "Pakistan", "Philippines", "Qatar", "Russian Federation", "Saudi Arabia", "Singapore",
            "Sri Lanka", "Syria", "Tajikistan", "Thailand", "Taiwan", "Turkey", "Turkmenistan", "United Arab Emirates",
            "Uzbekistan", "Vietnam", "Yemen"}
    europe = {"Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium", "Bosnia and Herzegovina",
              "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Georgia",
              "Germany", "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia", "Liechtenstein",
              "Lithuania", "Luxembourg", "Macedonia", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands",
              "Norway", "Poland", "Portugal", "Romania", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain",
              "Sweden", "Switzerland", "Ukraine", "United Kingdom", "Vatican City"}
    north_am = {"Antigua and Barbuda", "Bahamas", "Barbados", "Belize", "Canada", "Costa Rica", "Cuba", "Dominica",
                "Dominican Republic", "El Salvador", "Grenada", "Guadeloupe", "Guatemala", "Haiti", "Honduras",
                "Jamaica", "Mexico", "Nicaragua", "Panama", "Puerto Rico", "Saint Kitts and Nevis", "Saint Lucia",
                "Saint Vincent and the Grenadines", "Trinidad and Tobago", "United States"}
    oceania = {"Australia", "Fiji", "French Polynesia", "Kiribati", "Marshall Islands", "Micronesia", "Nauru",
               "New Zealand", "Palau", "Papua New Guinea", "Samoa", "Solomon Islands", "Tonga", "Tuvalu", "Vanuatu"}
    south_am = {"Argentina", "Bolivia", "Brazil", "Chile", "Colombia", "Ecuador", "Guyana", "Paraguay", "Peru",
                "Suriname", "Uruguay", "Venezuela"}

    for i in range(0, l_data_range):
        countries = dict()
        by_cont = [0] * 7
        if logic == 'member':
            d1 = datetime.datetime(2016 + int(i / 12), i % 12 + 1, 1)
            d2 = datetime.datetime(2015 + int(i / 12), i % 12 + 1, 1)
        else:
            d1 = datetime.datetime(2016 + int(i / 12), i % 12 + 1, 1)
            d2 = datetime.datetime(2016 + int(i / 12), i % 12 + 1, 1)
        for m in l_members:
            tester = m[4] + m[5]
            if m[0] in ['Normal Member', 'Associate Member', 'Fee-waiver Member'] and m[1] and m[1] < d1 and d2 < m[2] \
                    and tester not in countries.get(m[6], []):
                countries[m[6]] = countries.get(m[6], set())
                countries[m[6]].add(tester)
                if not m[6]:
                    m[6] = 'unknown'

        for country in countries:
            if country in africa:
                by_cont[0] += len(countries[country])
            elif country in asia:
                by_cont[1] += len(countries[country])
            elif country in europe:
                by_cont[2] += len(countries[country])
            elif country in north_am:
                by_cont[3] += len(countries[country])
            elif country in oceania:
                by_cont[4] += len(countries[country])
            elif country in south_am:
                by_cont[5] += len(countries[country])
            else:
                by_cont[6] += len(countries[country])
        return_list.append([datetime.datetime(2016 + int(i / 12), i % 12 + 1, 1), by_cont[0], by_cont[1], by_cont[2],
                            by_cont[3], by_cont[4], by_cont[5], by_cont[6]])

    return return_list


if __name__ == '__main__':

    # Prepare data

    # Read data from CiviCRM, export there with
    # Reports -> All Reports -> Complete list of members active and non-active -> add Country to selection
    # Actions -> Export as CSV
    with open(input_file_location, encoding="utf8") as csvfile:
        spamreader = list(csv.reader(csvfile, delimiter=',', quotechar='"', skipinitialspace=True))

    # convert the csv file to a list of lists
    # [0] Membership Type,    [1] = Start Date,   [2] = End Date,     [3] = Join Date
    # [4] = E-Mail,           [5] = name,         [6] = country
    members = make_neat(spamreader)

    # list with numbers for each member category for each month
    # returns list with
    # [0] date   [1] No of normal Members    [2] No of Associate Members,
    # [3] No of Fee Waiver Member            [4] No of Corporate Member
    history = by_months(members, data_range)

    # creates dataframe with the of by_months ^^
    columns = 'date', 'normalMember', 'associateMember', 'feeWaiverMember', 'corporateMember'
    history_df = pd.DataFrame(history, columns=columns)
    # set date as index
    history_df = history_df.set_index(history_df.columns[0])
    # drop the column that holds the date as we have set it as index
    history_df = history_df.iloc[1:]

    # generate data for the time before CiviCRM
    # load file and make a dataframe out of it
    precivic_history_df = pd.read_excel(history_file_location)
    # set date column as index to prepare merger with CiviCRM data
    precivic_history_df = precivic_history_df.set_index(precivic_history_df.columns[0])
    # drop the column that holds the date as we have set it as index
    precivic_history_df = precivic_history_df.iloc[1:]

    # combinde the two dataframes i.e. from CiviCRM and from the Excel file loaded earlier
    total_history_df = pd.concat([history_df, precivic_history_df], sort=True)
    total_history_df.sort_index(inplace=True)

    # replace zeros with nan
    total_history_df.replace(0, np.nan, inplace=True)

    # calcuate members by continent
    by_continent = continent_history(members, data_range)
    # create dataframe with the results
    columns = 'date', 'Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania', 'unknown'
    history_bycontinent_df = pd.DataFrame(by_continent, columns=columns)
    # set date as index
    history_bycontinent_df = history_bycontinent_df.set_index(history_bycontinent_df.columns[0])
    # drop the column that holds the date as we have set it as index
    history_bycontinent_df = history_bycontinent_df.iloc[1:]

    # calculate members by country for last period
    members_by_country = by_country(members, data_range)

    # generate and save charts, print charts if headless != True
    # bar chart with OSMF members and mappers by country
    mapper_members_df = bar_print_by_country(members_by_country, data_range, mappers_file)
    # OSMF members per member type by month - line and stacked area chart
    print_member_history_chart(total_history_df)
    # OSMF members per continent by month - line and stacked area chart
    print_by_continent(history_bycontinent_df)
    # Pie chart with members per country
    pie_print_by_country(members_by_country, data_range)

    # Print data to console
    #
    print(history_bycontinent_df)
    print(total_history_df)
    print('Members ex Corp by coountry at ',
          datetime.datetime(2016 + int((data_range - 1) / 12), (data_range - 1) % 12 + 1, 1))
    for line in members_by_country:
        print(line[1], line[0])
    mapper_member_regression(mapper_members_df)
