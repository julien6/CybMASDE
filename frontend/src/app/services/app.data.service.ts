import { Injectable } from "@angular/core";
import { BehaviorSubject } from 'rxjs';

@Injectable()
export class DataService {

  private configurationData$ = new BehaviorSubject<any>({});
  currentConfigurationData$ = this.configurationData$.asObservable();

  private stateData$ = new BehaviorSubject<any>({});
  currentStateData$ = this.stateData$.asObservable();

  constructor() {
  }

  public setConfigurationData(configurationData: any) {
    this.configurationData$.next(configurationData)
  }

  public setStateData(stateData: any) {
    this.stateData$.next(stateData)
  }

}
