// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import info.bliki.wiki.model.WikiModel;


public class MediaWikiToHtml {
    public static String convert(String mediaWikiText){
        String htmlText = WikiModel.toHtml(mediaWikiText);

        return htmlText;
    }

    public static void main(String[] args) {
        System.out.println(convert("{| class=wikitable\n" +
                "|+ Voter Registration and Party Enrollment {{as of|2018|July|3|lc=y|df=US}}<ref>{{cite web|url=http://www.elections.alaska.gov/statistics/2018/JUL/VOTERS%20BY%20PARTY%20AND%20PRECINCT.htm#STATEWIDE |title=Alaska Voter Registration by Party/Precinct |publisher=Elections.alaska.gov |date= |accessdate=January 9, 2019}}</ref>\n" +
                "|-\n" +
                "! colspan = 2 | Party\n" +
                "! Number of Voters\n" +
                "! Percentage\n" +
                "|-\n" +
                "{{party color|Independent politician}}\n" +
                "| [[Independent voter|Unaffiliated]]\n" +
                "| style=\"text-align:center;\"| 299,365\n" +
                "| style=\"text-align:center;\"| 55.25%\n" +
                "|-\n" +
                "{{party color|Republican Party (United States)}}\n" +
                "| [[Republican Party (United States)|Republican]]\n" +
                "| style=\"text-align:center;\"| 139,615\n" +
                "| style=\"text-align:center;\"| 25.77%\n" +
                "|-\n" +
                "{{party color|Democratic Party (United States)}}\n" +
                "|[[Democratic Party (United States)|Democratic]]\n" +
                "| style=\"text-align:center;\"| 74,865\n" +
                "| style=\"text-align:center;\"| 13.82%\n" +
                "|-\n" +
                "{{party color|Alaskan Independence Party}}\n" +
                "|[[Alaskan Independence Party|AKIP]]\n" +
                "| style=\"text-align:center;\"| 17,118\n" +
                "| style=\"text-align:center;\"| 3.16%\n" +
                "|-\n" +
                "{{party color|Libertarian Party (United States)}}\n" +
                "|[[Libertarian Party (United States)|Libertarian]]\n" +
                "| style=\"text-align:center;\"| 7,422\n" +
                "| style=\"text-align:center;\"| 1.37%\n" +
                "|-\n" +
                "{{party color|Independent (politician)}}\n" +
                "|[[List of political parties in the United States|Other]]\n" +
                "| style=\"text-align:center;\"| 3,436\n" +
                "| style=\"text-align:center;\"| 0.36%\n" +
                "|-\n" +
                "! colspan=\"2\" | Total\n" +
                "! style=\"text-align:center;\" | 541,821\n" +
                "! style=\"text-align:center;\" | 100%\n" +
                "|}"));
    }
}
