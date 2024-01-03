import { Component, OnInit } from '@angular/core';
import { DataService } from '../services/app.data.service';

@Component({
  selector: 'app-agents-logs',
  templateUrl: './agents-logs.component.html',
  styleUrls: ['./agents-logs.component.css']
})
export class AgentsLogsComponent implements OnInit {
  logsData: string;
  teams: any;

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
            const cyborgObservations = data["cyborg_observations"]

            for(const team in teamToAgents) {
              for(const agent of teamToAgents[team]){
                if(this.teams[team.toLowerCase()][agent] == null || this.teams[team.toLowerCase()][agent] == undefined){
                  this.teams[team.toLowerCase()][agent] = []
                }
                this.teams[team.toLowerCase()][agent].push({
                  episode_number: episodeNumber,
                  iteration_number: iterationNumber,
                  observations: cyborgObservations[agent],
                  actions: cyborgActions[agent]
                })
              }
            }
          }
        }
      }
    })

  }

}
