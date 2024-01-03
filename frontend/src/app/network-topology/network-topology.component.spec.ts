import { ComponentFixture, TestBed } from '@angular/core/testing';

import { NetworkTopologyComponent } from './network-topology.component';

describe('NetworkTopologyComponent', () => {
  let component: NetworkTopologyComponent;
  let fixture: ComponentFixture<NetworkTopologyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ NetworkTopologyComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(NetworkTopologyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
