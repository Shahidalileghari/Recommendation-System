def catalog_coverage(recommendation_lists, catalog_items) -> float:
    catalog = set(catalog_items)
    if not catalog:
        return 0.0
    recommended = {item for items in recommendation_lists for item in items}
    return len(recommended & catalog) / len(catalog)
