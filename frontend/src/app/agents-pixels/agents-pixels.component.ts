import { Component, OnInit } from '@angular/core';
import { DataService } from '../services/app.data.service';

@Component({
  selector: 'app-agents-pixels',
  templateUrl: './agents-pixels.component.html',
  styleUrls: ['./agents-pixels.component.css']
})
export class AgentsPixelsComponent implements OnInit {

  public _data: any = {}

  constructor(private dataService: DataService) {
  }

  ngOnInit() {

    this._data = {
      "green": {},
      "blue": {},
      "red": {}
    }

    this.dataService.currentStateData$.subscribe((data) => {
      if (data !== null && data !== undefined) {
        if (typeof data == 'object') {
          if (Object.keys(data).length > 0) {
            const agentsActions = data["agents_actions"]
            const agentObservations = data["agents_observations"]
            const actionSpaces = data["action_spaces"]
            const observationSpaces = data["observation_spaces"]

            for (const team in data["team_agent_mapping"]) {
              for (const agentName of data["team_agent_mapping"][team]) {
                if (Object.keys(data["agents_actions"]).includes(agentName)) {
                  if (this._data[team.toLowerCase()][agentName] == undefined || this._data[team.toLowerCase()][agentName] == null) {
                    this._data[team.toLowerCase()][agentName] = []
                  }
                  this._data[team.toLowerCase()][agentName].push({
                    observations: [0, 1, 2, 10],
                    actions: [agentsActions[agentName]]
                  })
                }
              }
            }

          }
        }
      }
    })


  }


}
