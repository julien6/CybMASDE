import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable()
export class ConfigShareService {

  private initialConfiguration = {
    "show_hide": {
      "prompt": false,
      "properties": true,
      "graphical": true
    }
  }

  private configuration$ = new BehaviorSubject<any>(this.initialConfiguration);
  currentConfiguration$ = this.configuration$.asObservable();

  constructor() { }

  public getInitialConfiguration() {
    return this.initialConfiguration;
  }

  public setConfiguration(configuration: any) {
    this.configuration$.next(configuration);
  }

  public hideShow(component: string, value: boolean) {

    let config = this.configuration$.getValue()
    config["show_hide"][component] = value;
    this.configuration$.next(config)

  }

}
