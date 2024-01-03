import { Component, OnInit, QueryList, ViewChildren } from '@angular/core';
import { ChartOptions } from 'chart.js';
import { BaseChartDirective } from 'ng2-charts';
import { DataService } from '../services/app.data.service';

@Component({
  selector: 'app-agents-piecharts',
  templateUrl: './agents-piecharts.component.html',
  styleUrls: ['./agents-piecharts.component.css']
})
export class AgentsPiechartsComponent implements OnInit {

  @ViewChildren(BaseChartDirective) public piecharts: QueryList<BaseChartDirective>;

  teams: any
  lastEpisode: number = -1;

  // Pie
  public pieChartOptions: ChartOptions<'pie'> = {
    responsive: false,
  };

  public pieChartLegend = false;
  public pieChartPlugins = [];

  constructor(private dataService: DataService) {

    this.teams = {
      "green": {},
      "red": {},
      "blue": {}
    }

  }

  ngOnInit(): void {
    this.dataService.currentStateData$.subscribe((data) => {
      if (data !== null && data !== undefined) {
        if (typeof data == 'object') {
          if (Object.keys(data).length > 0) {
            const teamToAgents = data["team_agent_mapping"]
            const episodeNumber = data["episode_number"]
            const iterationNumber = data["iteration_number"]
            const cyborgActions = data["cyborg_actions"]
            const agentsActions = data["agents_actions"]
            const cyborgObservations = data["cyborg_observations"]
            const pz_cyborg_actions = data["pz_cyborg_actions"]
            const activeAgents = Object.keys(cyborgObservations)
            if (this.lastEpisode !== episodeNumber) {

              for (const team in teamToAgents) {

                for (const agent of teamToAgents[team]) {

                  if (activeAgents.includes(agent)) {

                    // Adding a new data frame for new episode

                    if (this.teams[team.toLowerCase()][agent] == null || this.teams[team.toLowerCase()][agent] == undefined) {
                      // If first episode adding all agents
                      this.teams[team.toLowerCase()][agent] = {
                        "labels": [],
                        "datasets": []
                      }
                    }

                    const actionNames = []
                    const actionFrequencies = []
                    for (const action in pz_cyborg_actions[agent]) {
                      const actionName = action
                      actionNames.push(actionName)
                      actionFrequencies.push(0)
                    }
                    this.teams[team.toLowerCase()][agent]["labels"] = actionNames

                    this.teams[team.toLowerCase()][agent]["datasets"].push({
                      data: actionFrequencies
                    })
                  }
                }
              }

            } else {
              // Updating data for current episode
              for (const team in teamToAgents) {
                for (const agent of teamToAgents[team]) {
                  if (activeAgents.includes(agent)) {

                    // Adding a new data frame for new episode

                    const currentActFreqs = this.teams[team.toLowerCase()][agent]["datasets"][episodeNumber].data

                    const playedAction = JSON.stringify(agentsActions[agent])
                    for (const actionIndex in this.teams[team.toLowerCase()][agent]["labels"]) {
                      if (playedAction == this.teams[team.toLowerCase()][agent]["labels"][actionIndex]) {
                        currentActFreqs[actionIndex] = currentActFreqs[actionIndex] + 1
                        break
                      }
                    }

                    this.teams[team.toLowerCase()][agent]["datasets"][episodeNumber].data = currentActFreqs
                  }
                }
              }

            }
            this.lastEpisode = episodeNumber;
            this.piecharts.toArray().forEach((piechart) => {
              piechart.chart?.update('none')
            })

          }
        }
      }

    })
  }

}
