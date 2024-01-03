import { Component, OnInit, ViewChild, ElementRef } from '@angular/core';
import mermaid from 'mermaid';
import { DataService } from '../services/app.data.service';

@Component({
  selector: 'app-agents-seqdiagrams',
  templateUrl: './agents-seqdiagrams.component.html',
  styleUrls: ['./agents-seqdiagrams.component.css']
})
export class AgentsSeqdiagramsComponent implements OnInit {

  @ViewChild('mermaidDivGreen', { static: false }) mermaidDivGreen: ElementRef;
  @ViewChild('mermaidDivRed', { static: false }) mermaidDivRed: ElementRef;
  @ViewChild('mermaidDivBlue', { static: false }) mermaidDivBlue: ElementRef;

  graphDefs: any = {
    "blue": ["sequenceDiagram", "    A->>B: None"],
    "red": ["sequenceDiagram", "    A->>B: None"],
    "green": ["sequenceDiagram", "    A->>B: None"]
  }

  public ngAfterViewInit(): void {

    this.updateSeq();

  }

  ngOnInit(): void {

    mermaid.initialize({
      securityLevel: 'loose'
    });
    mermaid.init();

    this.dataService.currentStateData$.subscribe((data) => {
      if (data !== null && data !== undefined) {
        if (typeof data == 'object') {
          if (Object.keys(data).length > 0) {

            const episodeNumber = data["episode_number"]
            const iterationNumber = data["iteration_number"]

            if (episodeNumber == 0 && iterationNumber == 0) {
              return;
            }

            const teamToAgents = data["team_agent_mapping"]
            const agentsPzComms = data["agents_pz_comms"]
            const activeAgents = Object.keys(agentsPzComms)

            for (const team in teamToAgents) {

              for (const receiverAgent of teamToAgents[team]) {

                if (activeAgents.includes(receiverAgent)) {
                  if (this.graphDefs[team.toLowerCase()].includes("    A->>B: None")) {
                    const index = this.graphDefs[team.toLowerCase()].indexOf("    A->>B: None", 0);
                    if (index > -1) {
                      this.graphDefs[team.toLowerCase()].splice(index, 1);
                    }
                  }

                  for (const senderAgent in agentsPzComms[receiverAgent]) {
                    if ((senderAgent !== receiverAgent) && (agentsPzComms[receiverAgent][senderAgent] !== null)) {
                      this.graphDefs[team.toLowerCase()].push("    " + senderAgent + "->>" + receiverAgent
                        + ":" + agentsPzComms[receiverAgent][senderAgent].toString())
                      this.graphDefs[team.toLowerCase()].push("    " + "Note right of " + senderAgent + ": Episode "
                      + episodeNumber + " - Iteration " + iterationNumber)
                    }
                  }
                }

              }
            }
            this.updateSeq()
          }
        }
      }

    })

  }

  constructor(private dataService: DataService) {
  }

  updateSeq() {

    const elementGreen: any = this.mermaidDivGreen.nativeElement;
    const elementRed: any = this.mermaidDivRed.nativeElement;
    const elementBlue: any = this.mermaidDivBlue.nativeElement;

    mermaid.render('graphDivGreen', this.graphDefs.green.join("\n"), (svgCode, bindFunctionsGreen) => {
      elementGreen.innerHTML = svgCode;
      if (bindFunctionsGreen !== undefined) {
        bindFunctionsGreen(elementGreen);
      }
    });

    mermaid.render('graphDivRed', this.graphDefs.red.join("\n"), (svgCode, bindFunctionsRed: any) => {
      elementRed.innerHTML = svgCode;
      if (bindFunctionsRed !== undefined) {
        bindFunctionsRed(elementRed);
      }
    });

    mermaid.render('graphDivBlue', this.graphDefs.blue.join("\n"), (svgCode, bindFunctionsBlue: any) => {
      elementBlue.innerHTML = svgCode;
      if (bindFunctionsBlue !== undefined) {
        bindFunctionsBlue(elementBlue);
      }
    });
  }

}
