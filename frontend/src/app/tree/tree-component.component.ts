import { Component, OnInit, Input } from '@angular/core';
import { FlatTreeControl } from '@angular/cdk/tree';
import { MatTreeFlatDataSource, MatTreeFlattener } from '@angular/material/tree';

interface Node {
  name: string;
  children?: Node[];
}

/** Flat node with expandable and level information */
interface FlatNode {
  expandable: boolean;
  name: string;
  level: number;
}

@Component({
  selector: 'app-tree',
  templateUrl: './tree-component.component.html',
  styleUrls: ['./tree-component.component.css']
})
export class TreeComponent implements OnInit {

  @Input() data: any;

  ngOnInit() {
    this.dataSource.data = this._fromRawToTree(this.data);
  }

  constructor() {
    //this.dataSource.data = TREE_DATA;
  }

  public _fromRawToTree(rawData: any): Node[] {

    if (typeof rawData == 'string' || typeof rawData == 'number') {
      return [{ name: rawData.toString() }]
    } else {

      if (typeof rawData == 'object') {

        let processedData: Node[] = []
        for (const key in rawData) {
          processedData.push({
            name: key,
            children: this._fromRawToTree(rawData[key])
          })
        }
        return processedData
      }

    }

    return []

  }

  private _transformer = (node: Node, level: number) => {
    return {
      expandable: !!node.children && node.children.length > 0,
      name: node.name,
      level: level,
    };
  }

  treeControl = new FlatTreeControl<FlatNode>(
    node => node.level, node => node.expandable);

  treeFlattener = new MatTreeFlattener(
    this._transformer, node => node.level, node => node.expandable, node => node.children);

  dataSource = new MatTreeFlatDataSource(this.treeControl, this.treeFlattener);

  hasChild = (_: number, node: FlatNode) => node.expandable;

}