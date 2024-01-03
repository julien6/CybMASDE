import { Component, ElementRef, OnInit, OnDestroy, ViewChild, Input, OnChanges } from '@angular/core';
import { Network } from 'vis';
import { DataService } from '../services/app.data.service';
import { AppNetworkService } from '../services/app.network.service';

@Component({
  selector: 'app-network-topology',
  templateUrl: './network-topology.component.html',
  styleUrls: ['./network-topology.component.css']
})
export class NetworkTopologyComponent implements OnInit {

  @Input() stateData: any = {}; // data coming from detail panel component

  @ViewChild('treeContainer', { static: true }) treeContainer: ElementRef;

  private network: Network;

  private shownTeam: any = {
    "blue": false,
    "green": false,
    "red": false
  }

  constructor(private appNetworkService: AppNetworkService, private el: ElementRef, private dataService: DataService) { }//, private dataService: DataService) { }

  public ngOnInit(): void {

    const networkOptions: any = this.appNetworkService.getNetworkOptions()
    networkOptions["height"] = ((window.innerHeight - 160)).toString()

    this.network = new Network(
      this.treeContainer.nativeElement,
      this.stateData,
      networkOptions
    );

    this.dataService.currentStateData$.subscribe((data) => {
      if (this.network !== undefined) {
        if (data !== null && data !== undefined) {
          if (typeof data == 'object') {
            if (Object.keys(data).length > 0) {
              this.stateData = data
              this.network.setData(this.prepareDataForGraph(this.stateData, this.shownTeam["green"], this.shownTeam["red"], this.shownTeam["blue"]));
              this.network.redraw()
            }
          }
        }
      }
    })

  }

  public showTeam(team: string) {
    if (Object.keys(this.stateData).length > 0) {
      this.shownTeam[team] = !this.shownTeam[team]
      this.network.setData(this.prepareDataForGraph(this.stateData, this.shownTeam["green"], this.shownTeam["red"], this.shownTeam["blue"]));
      this.network.redraw()
    }
  }

  public prepareDataForGraph(data: any, showGreen: boolean = false, showRed: boolean = false, showBlue: boolean = false) {
    const networkGraph = data["network_graph"]

    const res: any = {
      "nodes": [],
      "edges": []
    }

    // Adding host nodes
    for (const node_id in networkGraph["nodes"]) {
      res["nodes"].push({ "id": networkGraph["nodes"][node_id]["id"], "label": networkGraph["nodes"][node_id]["id"] })
    }
    for (const link_id in networkGraph["links"]) {
      if (networkGraph["links"][link_id]["target"] !== networkGraph["links"][link_id]["source"]) {
        res["edges"].push({
          "from": networkGraph["links"][link_id]["source"],
          "to": networkGraph["links"][link_id]["target"],
          "id": networkGraph["links"][link_id]["source"] + "-" + networkGraph["links"][link_id]["target"]
        })
      }
    }

    const findTeam = (agent_name: string, mapping: any): string => {
      for (const team of Object.keys(mapping)) {
        if (mapping[team].includes(agent_name)) {
          return team
        }
      }
      return "None"
    }

    // Adding agents
    const team_agent_mapping: any = data["team_agent_mapping"]
    const shownTeams = [showBlue ? "blue" : "", showGreen ? "green" : "", showRed ? "red" : ""]
    for (const node of Object.keys(data["true_states"])) {
      if (node !== "success") {
        for (const session of Object.keys(data["true_states"][node]["Sessions"])) {
          const sess = data["true_states"][node]["Sessions"][session]
          const agent_name = sess["Agent"]
          const agent_team = findTeam(agent_name, team_agent_mapping).toLowerCase()
          if (agent_team !== "None") {
            if (shownTeams.includes(agent_team)) {
              const color = agent_team == "blue" ? [0, 0, 255] : agent_team == "red" ? [255, 0, 0] : agent_team == "green" ? [0, 255, 0] : [255, 255, 0]
              let lightFactor = 100
              const lightColor = agent_team == "blue" ? [lightFactor, lightFactor, 255] : agent_team == "red" ? [255, lightFactor, lightFactor] : agent_team == "green" ? [lightFactor, 255, lightFactor] : [255, 255, 0]
              lightFactor = 140
              const lightColor2 = agent_team == "blue" ? [lightFactor, lightFactor, 255] : agent_team == "red" ? [255, lightFactor, lightFactor] : agent_team == "green" ? [lightFactor, 255, lightFactor] : [255, 255, 0]

              res["nodes"].push({
                "id": agent_team + ":" + agent_name, "label": agent_name,
                "borderWidth": 1,
                "color": {
                  "background": 'rgb(' + lightColor[0] + ',' + lightColor[1] + ',' + lightColor[2] + ')',
                  "border": 'black',
                  "highlight": {
                    "background": "rgba(" + lightColor2[0] + "," + lightColor2[1] + "," + lightColor2[2] + ")",
                    "border": "black",
                    "borderWidth": 1
                  }
                }
              })

              res["edges"].push({
                from: node,
                to: agent_team + ":" + agent_name,
                length: 0,
                hidden: false,
                label: "",
                color: {
                  color: agent_team,
                  highlight: "rgba(" + lightColor2[0] + "," + lightColor2[1] + "," + lightColor2[2] + ")",
                },
                // smooth: {
                //   roundness: accessibleNodes.length == 1 ? 1 : ((j * (1 / (accessibleNodes.length - 1))) - 0.5),
                //   type: 'curvedCW'
                // },
                font: {
                  size: 10,
                  align: "middle"
                },
                physics: true
              })

            }
          }
        }
      }
    }

    return res;
  }

  public ngOnDestroy(): void {
    if (this.network != null) this.network.destroy();
  }

}
