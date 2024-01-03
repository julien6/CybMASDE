import { Component, Input, OnChanges, OnInit, SimpleChanges, ViewChild } from '@angular/core';
import { ChartConfiguration, ChartOptions, Chart } from "chart.js";
import { BaseChartDirective } from 'ng2-charts';

@Component({
  selector: 'app-global-graphs',
  templateUrl: './global-graphs.component.html',
  styleUrls: ['./global-graphs.component.css']
})
export class GlobalGraphsComponent implements OnInit {

  @ViewChild(BaseChartDirective) charts: BaseChartDirective;

  @Input() lastMetrics: any = {}; // data coming from detail panel component

  public lineChartData: ChartConfiguration<'line'>['data']

  public lineChartOptions: ChartOptions<'line'>

  public lineChartLegend: boolean

  constructor() {
    this.lineChartData = {
      labels: [0],
      datasets: [
        {
          data: [],
          label: 'Service Level Agreement',
          fill: true,
          tension: 0,
          borderColor: 'black',
          backgroundColor: 'rgba(255,0,0,0.3)'
        },
        {
          data: [],
          label: 'Total Value',
          fill: true,
          tension: 0,
          borderColor: 'black',
          backgroundColor: 'rgba(0,255,0,0.3)'
        }
      ]
    };

    this.lineChartOptions = {
      responsive: false
    };

    this.lineChartLegend = true;
  }


  ngOnInit(): void {
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (this.charts !== undefined) {
      if (this.charts !== null) {
        let labels = []
        for (let i = 1; i < this.lastMetrics["value"].length + 1; i++) {
          labels.push(i)
        }
        this.lineChartData.labels = labels
        this.lineChartData.datasets[1].data = this.lastMetrics["value"]
        this.charts.update()
      }
    }
  }

}
