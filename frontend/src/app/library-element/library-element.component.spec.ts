import { ComponentFixture, TestBed } from '@angular/core/testing';

import { LibraryElementComponent } from './library-element.component';

describe('LibraryElementComponent', () => {
  let component: LibraryElementComponent;
  let fixture: ComponentFixture<LibraryElementComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ LibraryElementComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(LibraryElementComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
