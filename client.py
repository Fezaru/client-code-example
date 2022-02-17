import logging
from datetime import date
from enum import Enum
from typing import List, Type, Union, Any, Tuple, Dict, Optional

import requests
from dateutil.relativedelta import relativedelta
from django.apps import apps
from django.core.exceptions import MultipleObjectsReturned
from django.db import models
from django.db.utils import IntegrityError
from requests.exceptions import Timeout
from retry import retry
from typing_extensions import Protocol
from xsdata.formats.dataclass.context import XmlContext
from xsdata.formats.dataclass.parsers import XmlParser

from portfolio_manager import dataclasses_pm


class Dataclass(Protocol):
  # the most reliable way to ascertain that something is a dataclass
  __dataclass_fields__: Dict


ENERGY_STAR_METRICS = ('abilityToShareForward', 'coolingDegreeDaysCDD', 'energyStarCertificationEligibility', 'groups',
                       'heatingDegreeDaysHDD', 'indirectGHGEmissions', 'institutionalPropertyYN',
                       # 'largestPropertyUseType', 'largestPropertyUseTypeGFA', 'lastModifiedByElectricMeters',
                       # 'lastModifiedByGasMeters', 'lastModifiedByOtherNonElectricNonGasEnergyMeters',
                       # 'lastModifiedByProperty', 'lastModifiedByPropertyUse', 'lastModifiedByUseDetails',
                       # 'lastModifiedByWaterMeters', 'lastModifiedDateElectricMeters', 'lastModifiedDateGasMeters',
                       # 'lastModifiedDateNonElectricNonGasEnergyMeters', 'lastModifiedDateProperty',
                       # 'lastModifiedDatePropertyUse', 'lastModifiedDateUseDetails', 'lastModifiedDateWasteMeters',
                       # 'lastModifiedDateWaterMeters', 'lastModifiedbyWasteMeters',
                       'listOfAllPropertyUseTypesAtProperty', 'medianScore', 'myPermissions',
                       'nonTimeWeightedFloorArea', 'numberOfPeopleWithAccess', 'parentPropertyName', 'peopleWithAccess',
                       'propGrossFloorArea', 'propertyDataAdministrator', 'propertyDataAdministratorAccountId',
                       'propertyDataAdministratorEmail', 'propertyFloorAreaBuildingsAndParking',
                       'propertyFloorAreaParking', 'referencePropertyTypeMedian', 'score',
                       'secondLargestPropertyUseType', 'secondLargestPropertyUseTypeGFA', 'serviceAndProductProvider',
                       'sharedByContact', 'sharedByContactAccountId', 'siteIntensity', 'siteTotal', 'sourceTotal',
                       'sourceTotalWN', 'systemDefinedPropertyType', 'thirdLargestPropertyUseType',
                       'thirdLargestPropertyUseTypeGFA', 'thirdPartyCertification',
                       'thirdPartyCertificationDateAchieved', 'thirdPartyCertificationDateAnticipated',
                       'weatherStationId', 'weatherStationName')

FOREIGN_KEY_ID_FIELDS = {
  # 'account': 'account_id',
  'property': 'property_id',
  'meter': 'meter_id',
}

logger = logging.getLogger(__name__)


def get_instances_id(instance: models.Model):
  return instance.id


class BaseClient:
  def __init__(self, username, password):
    self._headers = {'Content-Type': 'application/xml'}
    self._base_url = 'https://portfoliomanager.energystar.gov/ws'
    self._username = username
    self._password = password
    self._parser = XmlParser(context=XmlContext())
    self._session = None

  @property
  def session(self):
    if not self._session:
      self._session = requests.Session()
      self._session.auth = (self._username, self._password)

    return self._session

  def convert_to_django_model_objects(self, dataclass_objects: Optional[List[Dataclass]]) -> Optional[
    List[models.Model]]:
    if dataclass_objects is None:
      return None
    django_model_objects = [self._to_django_model(instance) for instance in dataclass_objects]
    return django_model_objects

  def convert_to_dataclass_objects(self, xml_content: str) -> List[Dataclass]:
    parsed_dataclass = self._to_dataclass(xml_content)
    dataclass_objects = self._handle_multiple_instances(parsed_dataclass)
    return dataclass_objects

  def _handle_multiple_instances(self, dataclass: Dataclass) -> List[Dataclass]:
    """
    Handles the response from the API: if the API gives back an instance, this instance is being
    returned in a list. If API gives back a Response model with link for every instance in the response, a request
    is being performed to get every instance, then a list of instances is returned
    """
    if type(dataclass) is dataclasses_pm.Response:
      links = self._get_links_from_list_response(dataclass)
      objects = self._get_objects_from_links(links)
    else:
      objects = [dataclass]
    return objects

  @retry(tries=3, delay=3, logger=logger)
  def _get(self, url_path: str, params: dict = None, headers: dict = None, timeout: float = 7) -> Tuple[
    Optional[requests.Response], bool]:
    """
    Wrapper for get of the requests library.
    @param url_path: URL path for request.
    @param params: query parameters for request.
    @param headers: list of headers for request.
    @param timeout: number of seconds to set for a timeout.
    """
    session = self.session
    complete_headers = self._headers
    if headers:
      complete_headers.update(headers)
    url = self._base_url + url_path
    try:
      response = session.get(url=url, headers=complete_headers, params=params, timeout=timeout)
    except Timeout:
      logger.exception(f'Request to the url {url} timed out')
      return None, False

    success = self._is_response_valid(response.status_code)
    logger.info(f'Request [{url}]. Status Code [{response.status_code}]')
    return response, success

  def _is_response_valid(self, status_code: int) -> bool:
    return status_code not in [400, 401, 403]

  def _get_links_from_list_response(self, list_response: dataclasses_pm.Response) -> List:
    return list_response.links.link

  def _get_objects_from_links(self, links: List) -> List[Dataclass]:
    parsed_dataclasses = []
    kwargs = {}
    for link in links:
      if link.id:
        kwargs.update({'id': link.id})
      response, success = self._get(link.link)
      if success:
        parsed_dataclasses.append(self._to_dataclass(response.text, **kwargs))

    return parsed_dataclasses

  def _to_dataclass(self, xml_content: str, **kwargs) -> Dataclass:
    parsed_dataclass = self._parser.from_string(xml_content)
    for k, v in kwargs.items():
      setattr(parsed_dataclass, k, v)
    return parsed_dataclass

  def _to_django_model(self, dataclass: Dataclass) -> Optional[models.Model]:
    django_model = self._get_django_model_from_dataclass_name(dataclass)
    return self._create_django_model_instance(django_model, dataclass)

  def _get_django_model_from_dataclass_name(self, dataclass: Type[Dataclass],
                                            app_label: str = 'portfolio_manager') -> Type[models.Model]:
    model_name = self._get_class_name(dataclass)
    return apps.get_model(app_label=app_label, model_name=model_name)

  def _create_django_model_instance(self, django_model: Type[models.Model], dataclass: Dataclass) -> Optional[
    models.Model]:
    logger.info(f'Started creating django model instance for {type(dataclass).__name__}')
    field_names = [k for k in dataclass.__dict__.keys()]
    model_fields = django_model._meta.get_fields()
    filtered_fields = list(
      filter(lambda x: x.name in field_names or FOREIGN_KEY_ID_FIELDS.get(x.name) in field_names, model_fields))
    instance = self._handle_conversion(django_model, filtered_fields, dataclass)
    logger.info(f'Finished creating django model instance for {type(dataclass).__name__}')
    return instance

  def _handle_conversion(self, django_model: Type[models.Model], filtered_fields: list, dataclass: Dataclass) -> \
          Optional[models.Model]:
    """
    Update logic: If objects pk field is in the filtered_fields default update logic happens: Get object and update its
    fields if it exists, create if it doesn't.
    If object has no many relations, then it is compared by fields in kwargs and taken from db or created.
    If object has many relations, it is being compared by relations and kwargs, and if relations are same, object from
    db is taken, otherwise new object created.
    """
    kwargs, many_relations = self._get_kwargs_and_relations(dataclass, filtered_fields)
    pk_fields = [field for field in filtered_fields if field.primary_key]
    if len(pk_fields) > 0:
      return self._update_or_create_object_with_relations(django_model, kwargs, many_relations, pk_fields[0].name)

    filtered_instances = django_model.objects.filter(**kwargs)
    if filtered_instances and not any(isinstance(field, models.ManyToManyField) for field in filtered_fields):
      if len(filtered_instances) != 1:
        raise MultipleObjectsReturned()
      return filtered_instances[0]
    elif len(filtered_instances) == 0:
      return self._update_or_create_object_with_relations(django_model, kwargs, many_relations)

    instance = self._get_object_with_same_relations(filtered_instances, many_relations)

    return instance or self._update_or_create_object_with_relations(django_model, kwargs, many_relations)

  def _update_or_create_object_with_relations(self, django_model: Type[models.Model], kwargs: dict,
                                              many_relations: dict, pk_field_name: str = None) -> Optional[
    models.Model]:
    """
    Update or create object and set its many relations.
    """
    instance = None
    if pk_field_name:
      instance = django_model.objects.filter(pk=kwargs[pk_field_name]).first()
    if instance is None:
      instance = django_model(**kwargs)
      try:
        instance.save()
      except IntegrityError as e:
        logger.exception(f'instance [{instance}] failed to be saved: message [{e}]')
        return None
    for name, values in many_relations.items():
      attr = getattr(instance, name)
      attr.set(values)
    instance.save()
    return instance

  def _get_kwargs_and_relations(self, dataclass: Dataclass, filtered_fields: list) -> Tuple[dict, dict]:
    many_relations = {}
    kwargs = {}
    for field in filtered_fields:
      if field.name in FOREIGN_KEY_ID_FIELDS.keys() and type(field) is models.ForeignKey:
        id_field_name = FOREIGN_KEY_ID_FIELDS[field.name]
        kwargs.update({id_field_name: getattr(dataclass, id_field_name)})
      else:
        converted_field = self._convert_field(field, dataclass)
        if isinstance(field, models.ManyToManyField):
          many_relations.update({field.name: converted_field})
        else:
          kwargs.update({field.name: converted_field})
    return kwargs, many_relations

  def _get_object_with_same_relations(self, filtered_instances: list, many_relations: dict):
    """
    Compare object to those in db by relations.
    """
    for instance in filtered_instances:
      is_matching = True
      for name, values in many_relations.items():
        values_in_db = list(getattr(instance, name).all())
        if not values_in_db.sort(key=get_instances_id) == values.sort(key=get_instances_id):
          is_matching = False
          break

      if is_matching:
        return instance

    return None

  def _convert_field(self, field, dataclass: Type[Dataclass]) -> Union[list, models.Model, Any, None]:
    """
    Handles field conversion: If the field is not a complex type, its value is being returned. If the value
    is a relation, instance is being checked if it is an enum, else if it has id so it can be retrieved from the db, or
    else it is being created. Also ManyToOneRel is being skipped because that is a reversed relation field in the model
    that is not needed.
    """
    if type(field) is models.ManyToOneRel or getattr(dataclass, field.name) is None:
      return None
    related_dataclass_content = getattr(dataclass, field.name)

    if type(field) is models.ManyToManyField:
      related_model = field.related_model
      values = [self.handle_relation_conversion(instance, related_model) for instance in
                related_dataclass_content]
      return values

    elif type(field) is models.ForeignKey:
      related_model = field.related_model
      return self.handle_relation_conversion(related_dataclass_content, related_model)
    elif type(field) in [models.DateTimeField, models.DateField]:
      """
      XmlDate and XmlDateTime classes are being parsed to Date and DateTime accordingly.
      String 'System Determined' may come from the API as a value for these fields, so date_value is being checked
      to return null instead of string.
      """
      date_value = None
      if type(field) is models.DateTimeField:
        date_value = getattr(dataclass, field.name).to_datetime()
      elif type(field) is models.DateField:
        date_value = getattr(dataclass, field.name).to_date()
      elif isinstance(date_value, str):
        return None
      return date_value
    return getattr(dataclass, field.name)

  def handle_relation_conversion(self, related_dataclass_content, related_model: Type[models.Model]) -> Optional[
    Type[models.Model]]:
    if self._is_enum_element(related_dataclass_content):
      return related_model.objects.get(name=related_dataclass_content.value)
    return self._create_django_model_instance(related_model, related_dataclass_content)

  def _is_enum_element(self, dataclass: Dataclass) -> bool:
    """
    This function checks if the element is an instance of Enum class.
    Note: This function gets enum element as a parameter, like AgencyTypeCountry.US
    """
    return isinstance(dataclass, Enum)

  def _get_class_name(self, instance: Dataclass) -> str:
    # return type(instance).__qualname__.replace('.', '')
    return type(instance).__name__

  def _batch(self, list, n):
    for i in range(0, len(list), n):
      yield list[i:i + n]

  def get_converted_to_django_model_objects(self, url: str, params: dict = None, headers: dict = None) -> Optional[
    List[models.Model]]:
    dataclass_objects = self.get_converted_to_dataclass_objects(url, params, headers)
    return self.convert_to_django_model_objects(dataclass_objects)

  def get_converted_to_dataclass_objects(self, url: str, params: dict = None, headers: dict = None) -> Optional[
    List[Dataclass]]:
    response, success = self._get(url, params, headers)
    return self.convert_to_dataclass_objects(response.text) if success else None


class PortfolioManagerClient(BaseClient):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._customers = None
    self.METRICS_BATCH_SIZE = 10

  @property
  def customers(self) -> Optional[List[models.Model]]:
    if not self._customers:
      self._customers = self.load_customers()

    return self._customers

  def get_account_info(self) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects('/account')

  def load_account_info(self) -> Optional[List[models.Model]]:
    account_dataclasses = self.get_account_info()
    return self.convert_to_django_model_objects(account_dataclasses)

  def get_account_shares(self) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects('/connect/account/pending/list')

  def load_account_shares(self) -> Optional[List[models.Model]]:
    account_shares_dataclasses = self.get_account_shares()
    return self.convert_to_django_model_objects(account_shares_dataclasses)

  def get_property_shares(self) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects('/share/property/pending/list')

  def load_property_shares(self) -> Optional[List[models.Model]]:
    property_shares_dataclasses = self.get_property_shares()
    return self.convert_to_django_model_objects(property_shares_dataclasses)

  def get_meter_shares(self) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects('/share/meter/pending/list')

  def load_meter_shares(self) -> Optional[List[models.Model]]:
    meter_shares_dataclasses = self.get_meter_shares()
    return self.convert_to_django_model_objects(meter_shares_dataclasses)

  def get_portfolio_manager_notifications(self) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects('/notification/list')

  def load_portfolio_manager_notifications(self) -> Optional[List[models.Model]]:
    notifications_dataclass_intances = self.get_portfolio_manager_notifications()
    return self.convert_to_django_model_objects(notifications_dataclass_intances)

  def get_customers(self) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects('/customer/list')

  def load_customers(self) -> Optional[List[models.Model]]:
    customers_dataclasses = self.get_customers()
    return self.convert_to_django_model_objects(customers_dataclasses)

  def get_customer_properties(self, customer_account_id: int) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects(f'/account/{customer_account_id}/property/list')

  def load_customer_properties(self, customer_account_id: int) -> Optional[
    List[models.Model]]:
    properties_dataclasses = self.get_customer_properties(customer_account_id)
    return self.convert_to_django_model_objects(properties_dataclasses)

  def get_property(self, property_id: int) -> Optional[Dataclass]:
    response, success = self._get(f'/property/{property_id}')
    if not success:
      return None

    return self._to_dataclass(response.text, id=property_id)

  def load_property(self, property_id: int) -> Optional[models.Model]:
    property_dataclass = self.get_property(property_id)
    return self.convert_to_django_model_objects([property_dataclass])

  def get_properties(self, customer_id: int = None) -> Optional[List[Dataclass]]:
    if customer_id:
      return self.get_customer_properties(customer_id)

    properties_ids_list = set()

    for customer in self.customers:
      response, success = self._get(f'/account/{customer.id}/property/list')
      if success:
        properties_list = self._to_dataclass(response.text)
        properties_ids_list.update([link.id for link in properties_list.links.link])

    properties_dataclasses = []
    for property_id in properties_ids_list:
      property_dataclass = self.get_property(property_id)
      if property_dataclass is not None:
        properties_dataclasses.append(property_dataclass)

    return properties_dataclasses

  def load_properties(self, customer_id: int = None) -> Optional[List[models.Model]]:
    if customer_id:
      return self.load_customer_properties(customer_id)

    properties_dataclasses = self.get_properties(customer_id)
    return self.convert_to_django_model_objects(properties_dataclasses)

  def get_property_hierarchy(self, property_id: int) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects(f'/idHierarchy/property/{property_id}')

  def load_property_hierarchy(self, property_id: int) -> Optional[models.Model]:
    property_hierarchy = self.get_property_hierarchy(property_id)
    return self.convert_to_django_model_objects(property_hierarchy)

  def get_property_uses(self, property_id: int) -> Optional[List[Dataclass]]:
    property_uses = []
    response, success = self._get(f'/property/{property_id}/propertyUse/list')
    if success:
      property_uses_list_response = self._to_dataclass(response.text)
      for link in property_uses_list_response.links.link:
        property_use_dataclass = self.get_property_use(link.id)
        if property_use_dataclass is None:
          logger.warning(f'Property use {link.id} of property {property_id} failed to be converted')
          continue

        property_uses.append(property_use_dataclass)

      return property_uses

    return None

  def load_property_uses(self, property_id: int) -> Optional[List[models.Model]]:
    property_uses = self.get_property_uses(property_id)
    return self.convert_to_django_model_objects(property_uses)

  def get_meters_for_property(self, property_id: int, only_shared: bool = True) -> Optional[List[Dataclass]]:
    """
    only_shared: If True, return only the meters that are shared. If False, return all meters.
    Note: may cause errors because of 403 for unshared meters.
    """
    return self.get_converted_to_dataclass_objects(f'/property/{property_id}/meter/list?myAccessOnly={only_shared}')

  def load_meters_for_property(self, property_id: int, only_shared: bool = True) -> Optional[List[models.Model]]:
    meters_dataclasses = self.get_meters_for_property(property_id, only_shared)
    return self.convert_to_django_model_objects(meters_dataclasses)

  def load_metrics(self, property_id: int, metrics_date: date = None, metrics: list = ENERGY_STAR_METRICS,
                   include_not_valid: bool = True) -> List[models.Model]:
    property_metrics_dataclasses = self.get_metrics(property_id, metrics_date, metrics, include_not_valid)
    return self.convert_to_django_model_objects(property_metrics_dataclasses)

  def get_metrics(self, property_id: int, metrics_date: date = None, metrics: list = ENERGY_STAR_METRICS,
                  include_not_valid: bool = True) -> List[Dataclass]:
    if not metrics_date:
      metrics_date = date.today()
    year = metrics_date.year
    month = metrics_date.month
    property_metrics_dataclasses = []
    # Batch metrics to avoid asking for too many at once
    for metrics_batch in self._batch(metrics, n=self.METRICS_BATCH_SIZE):
      property_metric_dataclasses_batch = self.get_converted_to_dataclass_objects(
        url=f'/property/{property_id}/metrics',
        params={'year': year, 'month': month,
                'measurementSystem': 'EPA'},
        headers={
          'PM-Metrics': ','.join(metrics_batch)})
      if property_metric_dataclasses_batch is not None:
        for property_metric_dataclass_instance in property_metric_dataclasses_batch:
          if include_not_valid:
            property_metrics_dataclasses.append(property_metric_dataclass_instance)
          elif any(metric.value is not None for metric in property_metric_dataclass_instance.metric):
            property_metrics_dataclasses.append(property_metric_dataclass_instance)

    return property_metrics_dataclasses

  def load_metrics_for_period(self, property_id: int, start_date: date = date(1999, 1, 1), end_date: date = None,
                              metrics: list = ENERGY_STAR_METRICS, include_not_valid: bool = True) -> List[
    models.Model]:
    """
    Get specified metrics for specified period.
    start_date: Only the year and month attributes are considered.
    end_date: Only the year and month attributes are considered.
    """
    metrics_raw = self.get_metrics_for_period(property_id, start_date, end_date, metrics, include_not_valid)
    return self.convert_to_django_model_objects(metrics_raw[::-1])

  def get_metrics_for_period(self, property_id: int, start_date: date = date(1999, 1, 1), end_date: date = None,
                             metrics: list = ENERGY_STAR_METRICS, include_not_valid: bool = True) -> List[Dataclass]:
    if not end_date:
      end_date = date.today()
    result_metrics = []
    months_back = 0
    while end_date + relativedelta(months=-months_back) >= start_date:
      year = (end_date + relativedelta(months=-months_back)).year
      month = (end_date + relativedelta(months=-months_back)).month
      metrics_dataclass_object = self.get_metrics(property_id, date(year, month, 1), metrics)
      if metrics_dataclass_object is not None:
        for property_metric in metrics_dataclass_object:
          if include_not_valid:
            result_metrics.append(property_metric)
          elif any(metric.value is not None for metric in property_metric.metric):
            result_metrics.append(property_metric)

      months_back += 1

    return result_metrics

  def get_reasons_for_no_score(self, property_id: int) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects(f'/property/{property_id}/reasonsForNoScore')

  def load_reasons_for_no_score(self, property_id: int) -> Optional[List[models.Model]]:
    reasons_for_no_score_dataclasses = self.get_reasons_for_no_score(property_id)
    return self.convert_to_django_model_objects(reasons_for_no_score_dataclasses)

  def get_reasons_for_no_water_score(self, property_id: int) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects(f'/property/{property_id}/reasonsForNoWaterScore')

  def load_reasons_for_no_water_score(self, property_id: int) -> Optional[List[models.Model]]:
    reasons_for_no_water_score_dataclasses = self.get_reasons_for_no_water_score(property_id)
    return self.convert_to_django_model_objects(reasons_for_no_water_score_dataclasses)

  def get_property_use(self, property_use_id: int) -> Optional[Dataclass]:
    response, success = self._get(f'/propertyUse/{property_use_id}')
    if not success:
      return None

    return self._to_dataclass(response.text, id=property_use_id)

  def load_property_use(self, property_use_id: int) -> Optional[List[models.Model]]:
    property_use_dataclass = self.get_property_use(property_use_id)
    return self.convert_to_django_model_objects([property_use_dataclass])

  def get_property_use_hierarchy(self, property_use_id: int) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects(f'/idHierarchy/propertyUse/{property_use_id}')

  def load_property_use_hierarchy(self, property_use_id: int) -> Optional[List[models.Model]]:
    property_use_hierarchy_dataclasses = self.get_property_use_hierarchy(property_use_id)
    return self.convert_to_django_model_objects(property_use_hierarchy_dataclasses)

  def get_meter_associations_for_property_use(self, property_use_id: int) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects(f'/association/propertyUse/{property_use_id}/meter')

  def load_meter_associations_for_property_use(self, property_use_id: int) -> Optional[List[models.Model]]:
    associations_dataclasses = self.get_meter_associations_for_property_use(property_use_id)
    return self.convert_to_django_model_objects(associations_dataclasses)

  def get_meter(self, meter_id: int) -> Optional[Dataclass]:
    return self.get_converted_to_dataclass_objects(f'/meter/{meter_id}')[0]

  def load_meter(self, meter_id: int) -> Optional[models.Model]:
    meter_dataclass = self.get_meter(meter_id)
    return self.convert_to_django_model_objects([meter_dataclass])

  def get_meter_hierarchy(self, meter_id: int) -> Optional[List[Dataclass]]:
    return self.get_converted_to_dataclass_objects(f'/meter/{meter_id}')

  def load_meter_hierarchy(self, meter_id: int) -> Optional[List[models.Model]]:
    meter_hierarchy_dataclasses = self.get_meter_hierarchy(meter_id)
    return self.convert_to_django_model_objects(meter_hierarchy_dataclasses)

  def load_meter_consumption(self, meter_id: int, start_date: date = date(1999, 1, 1), end_date: date = None) -> \
          Optional[List[models.Model]]:
    if not end_date:
      end_date = date.today()
    dataclass_objects = self.get_meter_consumption(meter_id, start_date, end_date)
    return self.convert_to_django_model_objects(dataclass_objects) if dataclass_objects else None

  def get_meter_consumption(self, meter_id: int, start_date: date = date(1999, 1, 1), end_date: date = None) -> \
          Optional[List[Dataclass]]:
    """
    Retrieves consumption values for meters.
    Pagination is involved, each page holds up to 120 records.
    Initial call returns a list of 1 object with set of links for next pages.
    """
    if not end_date:
      end_date = date.today()
    str_start_date = start_date.strftime('%Y-%m-%d')
    str_end_date = end_date.strftime('%Y-%m-%d')
    params = {'startDate': str_start_date, 'endDate': str_end_date}

    consumption_data_dataclasses = self.get_converted_to_dataclass_objects(
      url=f'/meter/{meter_id}/consumptionData', params=params)
    if consumption_data_dataclasses:
      first_meter_data = consumption_data_dataclasses[0]
      dataclass_objects = [first_meter_data]

      for link in first_meter_data.links.link:
        if link.link_description == 'next page':
          dataclass_objects.extend(self.get_converted_to_dataclass_objects(url=link.link))

      return dataclass_objects

    return None
