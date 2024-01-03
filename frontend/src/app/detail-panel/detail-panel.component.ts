import { Component, OnInit, ViewChild } from '@angular/core';
import { JsonEditorOptions } from "@maaxgr/ang-jsoneditor";
import { DataService } from '../../app/services/app.data.service';
import { ConfigShareService } from '../services/config-share.service';

@Component({
  selector: 'app-detail-panel',
  templateUrl: './detail-panel.component.html',
  styleUrls: ['./detail-panel.component.css'],
})
export class DetailPanelComponent implements OnInit {

  public editorOptions: JsonEditorOptions;
  public sharedConfigurationData: any;
  public configurationData: any;
  public stateData: any;

  public menuBarConfiguration: any;

  // options = {
  //   templates: [
  //     {
  //       text: 'Node',
  //       title: 'Insert a Network Node',
  //       className: 'jsoneditor-type-object',
  //       field: 'NodeTemplate',
  //       value: {
  //         'id': "Undefined ID",
  //         'label': 'Undefined Name',
  //       }
  //     },
  //     {
  //       text: 'Edge',
  //       title: 'Insert a Network Edge',
  //       field: 'EdgeTemplate',
  //       value: {
  //         'id': "Undefined ID",
  //         'from': "Undefined Start Node",
  //         'to': "Undefined End Node",
  //       }
  //     }
  //   ]
  // }

  public figData1 = { "width": 640.0, "height": 480.0, "axes": [{ "bbox": [0.125, 0.10999999999999999, 0.775, 0.77], "xlim": [-2.0, 22.0], "ylim": [-0.1, 1.1], "xdomain": [-2.0, 22.0], "ydomain": [-0.1, 1.1], "xscale": "linear", "yscale": "linear", "axes": [{ "position": "bottom", "nticks": 7, "tickvalues": null, "tickformat_formatter": "", "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": { "gridOn": true, "color": "#EEEEEE", "dasharray": "none", "alpha": 1.0 }, "visible": true }, { "position": "left", "nticks": 8, "tickvalues": null, "tickformat_formatter": "", "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": { "gridOn": true, "color": "#EEEEEE", "dasharray": "none", "alpha": 1.0 }, "visible": true }], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el4056140274109911664", "lines": [{ "data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el4056140274096628560", "color": "#000000", "linewidth": 5.0, "dasharray": "none", "alpha": 0.5, "zorder": 2, "drawstyle": "default" }], "paths": [], "markers": [{ "data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "id": "el4056140274096628560pts", "facecolor": "#008000", "edgecolor": "#90EE90", "edgewidth": 10, "alpha": 0.5, "zorder": 2, "markerpath": [[[0.0, 10.0], [2.652031, 10.0], [5.195798707848534, 8.946336915882418], [7.0710678118654755, 7.0710678118654755], [8.946336915882418, 5.195798707848534], [10.0, 2.652031], [10.0, 0.0], [10.0, -2.652031], [8.946336915882418, -5.195798707848534], [7.0710678118654755, -7.0710678118654755], [5.195798707848534, -8.946336915882418], [2.652031, -10.0], [0.0, -10.0], [-2.652031, -10.0], [-5.195798707848534, -8.946336915882418], [-7.0710678118654755, -7.0710678118654755], [-8.946336915882418, -5.195798707848534], [-10.0, -2.652031], [-10.0, 0.0], [-10.0, 2.652031], [-8.946336915882418, 5.195798707848534], [-7.0710678118654755, 7.0710678118654755], [-5.195798707848534, 8.946336915882418], [-2.652031, 10.0], [0.0, 10.0]], ["M", "C", "C", "C", "C", "C", "C", "C", "C", "Z"]] }], "texts": [], "collections": [], "images": [], "sharex": [], "sharey": [] }], "data": { "data01": [[0.0, 0.4115941736206715], [1.0, 0.20219699323751072], [2.0, 0.1741135038137297], [3.0, 0.5089275697878123], [4.0, 0.889965897153537], [5.0, 0.8142298145910671], [6.0, 0.24778218460714618], [7.0, 0.25322087990692954], [8.0, 0.8894387353053428], [9.0, 0.5849641760372588], [10.0, 0.8407796425854553], [11.0, 0.08238150578220405], [12.0, 0.3669345044345548], [13.0, 0.4292712830883173], [14.0, 0.870875342495901], [15.0, 0.8250707991639138], [16.0, 0.007186167130482368], [17.0, 0.8662766646072589], [18.0, 0.3955155369019654], [19.0, 0.015488946043075824]] }, "id": "el4056140274246707392", "plugins": [{ "type": "reset" }, { "type": "zoom", "button": true, "enabled": false }, { "type": "boxzoom", "button": true, "enabled": false }] }
  public figData2 = {"width": 640.0, "height": 480.0, "axes": [{"bbox": [0.125, 0.10999999999999999, 0.775, 0.77], "xlim": [-3.262615958401179, 3.3201380578060817], "ylim": [0.0, 92.4], "xdomain": [-3.262615958401179, 3.3201380578060817], "ydomain": [0.0, 92.4], "xscale": "linear", "yscale": "linear", "axes": [{"position": "bottom", "nticks": 9, "tickvalues": null, "tickformat_formatter": "", "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": true, "color": "#FFFFFF", "dasharray": "none", "alpha": 1.0}, "visible": true}, {"position": "left", "nticks": 6, "tickvalues": null, "tickformat_formatter": "", "tickformat": null, "scale": "linear", "fontsize": 10.0, "grid": {"gridOn": true, "color": "#FFFFFF", "dasharray": "none", "alpha": 1.0}, "visible": true}], "axesbg": "#FFFFFF", "axesbgalpha": null, "zoomable": true, "id": "el9177140694819224064", "lines": [], "paths": [{"data": "data01", "xindex": 0, "yindex": 1, "coordinates": "data", "pathcodes": ["M", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "L", "Z"], "id": "el9177140694803996800", "dasharray": "none", "alpha": 0.5, "facecolor": "rgba(173, 216, 230, 0.5)", "edgecolor": "none", "edgewidth": 1.0, "zorder": 1}], "markers": [], "texts": [], "collections": [], "images": [], "sharex": [], "sharey": []}], "data": {"data01": [[-2.9633998667553945, 0.0], [-2.9633998667553945, 3.0], [-2.763922472324871, 3.0], [-2.763922472324871, 2.0], [-2.5644450778943484, 2.0], [-2.5644450778943484, 8.0], [-2.364967683463825, 8.0], [-2.364967683463825, 7.0], [-2.1654902890333023, 7.0], [-2.1654902890333023, 5.0], [-1.966012894602779, 5.0], [-1.966012894602779, 20.0], [-1.766535500172256, 20.0], [-1.766535500172256, 22.0], [-1.567058105741733, 22.0], [-1.567058105741733, 32.0], [-1.36758071131121, 32.0], [-1.36758071131121, 39.0], [-1.168103316880687, 39.0], [-1.168103316880687, 39.0], [-0.9686259224501639, 39.0], [-0.9686259224501639, 45.0], [-0.7691485280196408, 45.0], [-0.7691485280196408, 59.0], [-0.5696711335891176, 59.0], [-0.5696711335891176, 67.0], [-0.37019373915859477, 67.0], [-0.37019373915859477, 80.0], [-0.1707163447280715, 80.0], [-0.1707163447280715, 88.0], [0.028761049702451302, 88.0], [0.028761049702451302, 83.0], [0.22823844413297456, 83.0], [0.22823844413297456, 70.0], [0.4277158385634978, 70.0], [0.4277158385634978, 63.0], [0.6271932329940206, 63.0], [0.6271932329940206, 55.0], [0.8266706274245439, 55.0], [0.8266706274245439, 54.0], [1.0261480218550667, 54.0], [1.0261480218550667, 39.0], [1.22562541628559, 39.0], [1.22562541628559, 31.0], [1.4251028107161128, 31.0], [1.4251028107161128, 34.0], [1.6245802051466356, 34.0], [1.6245802051466356, 21.0], [1.8240575995771593, 21.0], [1.8240575995771593, 9.0], [2.023534994007682, 9.0], [2.023534994007682, 8.0], [2.223012388438205, 8.0], [2.223012388438205, 8.0], [2.4224897828687286, 8.0], [2.4224897828687286, 3.0], [2.6219671772992514, 3.0], [2.6219671772992514, 4.0], [2.8214445717297743, 4.0], [2.8214445717297743, 2.0], [3.020921966160297, 2.0], [3.020921966160297, 0.0], [2.8214445717297743, 0.0], [2.8214445717297743, 0.0], [2.6219671772992514, 0.0], [2.6219671772992514, 0.0], [2.4224897828687286, 0.0], [2.4224897828687286, 0.0], [2.223012388438205, 0.0], [2.223012388438205, 0.0], [2.023534994007682, 0.0], [2.023534994007682, 0.0], [1.8240575995771593, 0.0], [1.8240575995771593, 0.0], [1.6245802051466356, 0.0], [1.6245802051466356, 0.0], [1.4251028107161128, 0.0], [1.4251028107161128, 0.0], [1.22562541628559, 0.0], [1.22562541628559, 0.0], [1.0261480218550667, 0.0], [1.0261480218550667, 0.0], [0.8266706274245439, 0.0], [0.8266706274245439, 0.0], [0.6271932329940206, 0.0], [0.6271932329940206, 0.0], [0.4277158385634978, 0.0], [0.4277158385634978, 0.0], [0.22823844413297456, 0.0], [0.22823844413297456, 0.0], [0.028761049702451302, 0.0], [0.028761049702451302, 0.0], [-0.1707163447280715, 0.0], [-0.1707163447280715, 0.0], [-0.37019373915859477, 0.0], [-0.37019373915859477, 0.0], [-0.5696711335891176, 0.0], [-0.5696711335891176, 0.0], [-0.7691485280196408, 0.0], [-0.7691485280196408, 0.0], [-0.9686259224501639, 0.0], [-0.9686259224501639, 0.0], [-1.168103316880687, 0.0], [-1.168103316880687, 0.0], [-1.36758071131121, 0.0], [-1.36758071131121, 0.0], [-1.567058105741733, 0.0], [-1.567058105741733, 0.0], [-1.766535500172256, 0.0], [-1.766535500172256, 0.0], [-1.966012894602779, 0.0], [-1.966012894602779, 0.0], [-2.1654902890333023, 0.0], [-2.1654902890333023, 0.0], [-2.364967683463825, 0.0], [-2.364967683463825, 0.0], [-2.5644450778943484, 0.0], [-2.5644450778943484, 0.0], [-2.763922472324871, 0.0], [-2.763922472324871, 0.0]]}, "id": "el9177140694954272960", "plugins": [{"type": "reset"}, {"type": "zoom", "button": true, "enabled": false}, {"type": "boxzoom", "button": true, "enabled": false}]}


  constructor(private dataService: DataService, private configShareService: ConfigShareService) {
    this.editorOptions = new JsonEditorOptions();
    this.editorOptions.mode = 'code';
    this.editorOptions.mainMenuBar = true;
    this.editorOptions.language = "en";
    // (<any>this.editorOptions).templates = this.options.templates;
    // this.editorOptions.expandAll = true;

    this.configurationData = {};
    this.sharedConfigurationData = this.configurationData;

    this.menuBarConfiguration = this.configShareService.getInitialConfiguration();

  }

  ngOnInit(): void {

    this.configShareService.currentConfiguration$.subscribe((data) => {
      this.menuBarConfiguration = data;
    })

    this.dataService.currentConfigurationData$.subscribe((data) => {
      this.sharedConfigurationData = data;
      this.configurationData = data;
    })

    this.dataService.currentStateData$.subscribe((data) => {
      this.stateData = data;
    })

  }

  showJson(d: Event) {
    this.sharedConfigurationData = d;
  }

}
