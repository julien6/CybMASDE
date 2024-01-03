import { Component, OnInit, QueryList, ViewChild, ViewChildren } from '@angular/core';
import { ChartConfiguration } from 'chart.js';
import { BaseChartDirective } from 'ng2-charts'
import { DataService } from '../services/app.data.service';

@Component({
  selector: 'app-agents-histograms',
  templateUrl: './agents-histograms.component.html',
  styleUrls: ['./agents-histograms.component.css']
})
export class AgentsHistogramsComponent implements OnInit {

  @ViewChildren(BaseChartDirective) public histograms: QueryList<BaseChartDirective>;

  public teams: any;

  private colorByEp: any[] = [];
  private lastEpisode: number = -1;

  public barChartLegend = true;
  public barChartPlugins = [];

  public barChartOptions: ChartConfiguration<'bar'>['options'] = {
    responsive: false,
    plugins: {
      legend: {
        display: true,
      },
      tooltip: {
        enabled: true
      }
    }
  };

  constructor(private dataService: DataService) {

    this.teams = {
      "green": {},
      "red": {},
      "blue": {}
    }
  }


  private pickUniqueRdmCol() {
    let col = null
    while (true) {
      const red = Math.ceil(Math.random() * 256)
      const green = Math.ceil(Math.random() * 256)
      const blue = Math.ceil(Math.random() * 256)
      col = [red, green, blue]
      if (!this.colorByEp.includes(col)) {
        break;
      }
    }
    this.colorByEp.push(col)
    return col
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

                    const color = this.pickUniqueRdmCol()
                    this.teams[team.toLowerCase()][agent]["datasets"].push({
                      data: actionFrequencies,
                      label: "Ep. " + episodeNumber,
                      backgroundColor: 'rgba(' + color[0] + ',' + color[1] + ',' + color[2] + ', 0.2)',
                      borderColor: 'rgb(' + color[0] + ',' + color[1] + ',' + color[2] + ')',
                      borderWidth: 1
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
            this.histograms.toArray().forEach((histogram) => {
              histogram.chart?.update('none')
            })

          }
        }
      }

    })
  }

}
