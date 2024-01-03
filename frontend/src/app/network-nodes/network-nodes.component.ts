import { Component, OnInit } from '@angular/core';
import {MatTableModule} from '@angular/material/table';
import { DataService } from '../services/app.data.service';

@Component({
  selector: 'app-network-nodes',
  templateUrl: './network-nodes.component.html',
  styleUrls: ['./network-nodes.component.css']
})
export class NetworkNodesComponent implements OnInit {

  logsData: string;
  nodesData: any

  constructor(private dataService: DataService) { }//, private dataService: DataService) { }

  public ngOnInit(): void {

    this.nodesData = {}

    this.dataService.currentStateData$.subscribe((data) => {
        if (data !== null && data !== undefined) {
          if (typeof data == 'object') {
            if (Object.keys(data).length > 0) {
              this.nodesData = data["true_states"]
            }
          }
        }
    })

  }


}
