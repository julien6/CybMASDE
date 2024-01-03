import { Injectable } from "@angular/core";
import { Options, DataSet } from "vis";

@Injectable()
export class AppNetworkService {
  public getNetworkOptions(): Options {
    return {
      autoResize: true,
      height: "800px",
      width: "100%",
      physics: true,
      nodes: {
        color: {
          background: "#8c8f8c",
          border: "grey",
          highlight: { background: "#c2c4c2", border: "#acb5ac" }
        },
        shape: "box"
      }
    };
  }
}
